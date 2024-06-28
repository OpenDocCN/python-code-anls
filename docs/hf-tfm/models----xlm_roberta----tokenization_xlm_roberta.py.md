# `.\models\xlm_roberta\tokenization_xlm_roberta.py`

```
# coding=utf-8
# 指定文件编码为 UTF-8

# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# 版权声明，列出了部分作者和团队

# Licensed under the Apache License, Version 2.0 (the "License");
# 遵循 Apache 2.0 许可证

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件。

# You may obtain a copy of the License at
# 可以从上述链接获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     许可证详细信息网址

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则按原样提供分发，无论是明示还是暗示的任何保证或条件。

# See the License for the specific language governing permissions and
# limitations under the License
# 请参阅许可证了解具体的权限和限制。

""" Tokenization classes for XLM-RoBERTa model."""
# XLM-RoBERTa 模型的分词类定义

import os
# 导入操作系统相关模块
from shutil import copyfile
# 导入文件复制功能模块
from typing import Any, Dict, List, Optional, Tuple
# 导入类型提示相关模块

import sentencepiece as spm
# 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 tokenization_utils 中导入 AddedToken 和 PreTrainedTokenizer 类
from ...utils import logging
# 从 utils 中导入 logging 模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

SPIECE_UNDERLINE = "▁"
# 定义特殊标记 "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}
# 词汇表文件名字典，包含一个键值对

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "FacebookAI/xlm-roberta-base": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/sentencepiece.bpe.model",
        "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/sentencepiece.bpe.model",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english/resolve/main/sentencepiece.bpe.model"
        ),
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german": (
            "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german/resolve/main/sentencepiece.bpe.model"
        ),
    }
}
# 预训练模型的词汇文件映射字典，包含多个模型名到 URL 的映射

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "FacebookAI/xlm-roberta-base": 512,
    "FacebookAI/xlm-roberta-large": 512,
    "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": 512,
    "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": 512,
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": 512,
    "FacebookAI/xlm-roberta-large-finetuned-conll03-german": 512,
}
# 预训练模型的位置嵌入大小字典，包含多个模型名到嵌入大小的映射

class XLMRobertaTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # XLM-RoBERTa 分词器类继承自 PreTrainedTokenizer 类
    # 定义函数，用于加载词汇表文件，并配置特殊标记的默认值
    def __init__(
        vocab_file: str,
        bos_token: str = "<s>",   # 开始序列的特殊标记，默认为 "<s>"
        eos_token: str = "</s>",   # 结束序列的特殊标记，默认为 "</s>"
        sep_token: str = "</s>",   # 分隔符的特殊标记，默认为 "</s>"
        cls_token: str = "<s>",    # 分类器标记，在使用特殊标记构建序列时为序列的第一个标记，默认为 "<s>"
        unk_token: str = "<unk>",  # 未知标记，当词汇表中不存在某个词时使用，默认为 "<unk>"
        pad_token: str = "<pad>",  # 填充标记，用于处理不同长度的序列，默认为 "<pad>"
        mask_token: str = "<mask>",# 掩码标记，用于掩盖值，模型训练中会预测该标记，默认为 "<mask>"
        sp_model_kwargs: dict = None # 传递给 SentencePieceProcessor.__init__() 方法的参数，用于配置 SentencePiece 模型
    ):
        """
        初始化函数，用于配置特殊标记的默认值和加载词汇表文件。
    
        Args:
            vocab_file (`str`): 词汇表文件的路径。
            bos_token (`str`, *optional*, defaults to `"<s>"`): 预训练期间使用的序列开始标记，也可用作序列分类器标记。
            eos_token (`str`, *optional*, defaults to `"</s>"`): 序列结束标记。
            sep_token (`str`, *optional*, defaults to `"</s>"`): 分隔符标记，用于构建多序列或特殊标记序列的最后一个标记。
            cls_token (`str`, *optional*, defaults to `"<s>"`): 序列分类时的分类器标记，用于整体序列分类。
            unk_token (`str`, *optional*, defaults to `"<unk>"`): 未知标记，用于词汇表中不存在的词。
            pad_token (`str`, *optional*, defaults to `"<pad>"`): 填充标记，用于处理不同长度序列的填充。
            mask_token (`str`, *optional*, defaults to `"<mask>"`): 掩码标记，模型预测时使用的标记。
            sp_model_kwargs (`dict`, *optional*): 将传递给 `SentencePieceProcessor.__init__()` 方法的参数。
        """
        pass
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        # 如果 mask_token 是字符串，则设置为 AddedToken 对象，它会去除左侧空格并被视为特殊标记
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则设为空字典，否则使用提供的参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 使用 SentencePieceProcessor 初始化 sp_model 对象，并加载给定的 vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # 确保 fairseq 和 spm 的词汇表对齐，以便进行 token-to-id 映射
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # fairseq 的偏移量，用于调整 fairseq 和 spm 的 token-to-id 映射关系
        self.fairseq_offset = 1

        # 添加 <mask> 到 token-to-id 映射中，使用 fairseq 的偏移量和 spm 的词汇表长度
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        # 创建 fairseq 的 id-to-token 映射
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法，传递所有参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def __getstate__(self):
        # 创建对象状态的拷贝
        state = self.__dict__.copy()
        # 将 sp_model 设为 None，以防止序列化时存储 sp_model 对象
        state["sp_model"] = None
        # 存储 sp_model 的序列化模型 proto
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        # 恢复对象状态
        self.__dict__ = d

        # 兼容旧版本的处理
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建 sp_model 对象，并从序列化的 proto 中加载模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
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

        # If only one sequence is provided, concatenate with special tokens <s> and </s>
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # Define special tokens for beginning (CLS) and separation (SEP)
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        
        # For a pair of sequences, concatenate with appropriate special tokens
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep


    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        # If the tokens already include special tokens, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # Initialize the mask with special token (1) for CLS token
        special_tokens_mask = [1]
        
        # Append sequence token (0) for each token in token_ids_0
        special_tokens_mask += [0] * len(token_ids_0)
        
        # Append special token (1) for SEP token
        special_tokens_mask += [1]
        
        # If token_ids_1 exists, append special tokens for separation and tokens in token_ids_1
        if token_ids_1 is not None:
            special_tokens_mask += [1, 1] + [0] * len(token_ids_1)
            special_tokens_mask += [1]
        
        return special_tokens_mask
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

        # Define the separator token
        sep = [self.sep_token_id]
        # Define the classification token
        cls = [self.cls_token_id]

        # If only one sequence is provided (no token_ids_1), return the mask length based on token_ids_0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided, return the mask length based on both sequences concatenated
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Calculate and return the vocabulary size including an offset for additional tokens (like <mask>)
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        # Generate a dictionary mapping tokens to their corresponding IDs in the vocabulary
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # Update the vocabulary with any additional tokens not in the SentencePiece model
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize input text into a list of subword strings using SentencePiece model
        # TODO check if the t5/llama PR also applies here
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        # Check if the token exists in the fairseq vocabulary mappings
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # If not found, convert using the SentencePiece model
        spm_id = self.sp_model.PieceToId(token)

        # Return an unknown token ID if SentencePiece model returns 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an ID (integer) into a token (str) using the vocabulary."""
        # Check if the ID exists in fairseq mappings
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # Convert ID to a token using the SentencePiece model, adjusting for fairseq offset
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string."""
        # Concatenate tokens into a string, replacing SPIECE_UNDERLINE with spaces and stripping extra spaces
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Ensure save_directory exists and is a directory; otherwise, log an error
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocabulary file differs from the output path and exists, copy it
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If the current vocabulary file doesn't exist, write the serialized SentencePiece model to the output file
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```