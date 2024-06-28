# `.\models\albert\tokenization_albert_fast.py`

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


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging


if is_sentencepiece_available():
    # 如果存在 sentencepiece 库，导入 AlbertTokenizer
    from .tokenization_albert import AlbertTokenizer
else:
    AlbertTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)
# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

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
    },
    # tokenizer_file 对于所有预训练模型都是 tokenizer.json
    "tokenizer_file": {
        "albert/albert-base-v1": "https://huggingface.co/albert/albert-base-v1/resolve/main/tokenizer.json",
        "albert/albert-large-v1": "https://huggingface.co/albert/albert-large-v1/resolve/main/tokenizer.json",
        "albert/albert-xlarge-v1": "https://huggingface.co/albert/albert-xlarge-v1/resolve/main/tokenizer.json",
        "albert/albert-xxlarge-v1": "https://huggingface.co/albert/albert-xxlarge-v1/resolve/main/tokenizer.json",
        "albert/albert-base-v2": "https://huggingface.co/albert/albert-base-v2/resolve/main/tokenizer.json",
        "albert/albert-large-v2": "https://huggingface.co/albert/albert-large-v2/resolve/main/tokenizer.json",
        "albert/albert-xlarge-v2": "https://huggingface.co/albert/albert-xlarge-v2/resolve/main/tokenizer.json",
        "albert/albert-xxlarge-v2": "https://huggingface.co/albert/albert-xxlarge-v2/resolve/main/tokenizer.json",
    },
}
    # 定义一个字典，包含不同版本的ALBERT模型名称和对应的分词器文件的URL
    "tokenizer_file": {
        "albert/albert-base-v1": "https://huggingface.co/albert/albert-base-v1/resolve/main/tokenizer.json",
        "albert/albert-large-v1": "https://huggingface.co/albert/albert-large-v1/resolve/main/tokenizer.json",
        "albert/albert-xlarge-v1": "https://huggingface.co/albert/albert-xlarge-v1/resolve/main/tokenizer.json",
        "albert/albert-xxlarge-v1": "https://huggingface.co/albert/albert-xxlarge-v1/resolve/main/tokenizer.json",
        "albert/albert-base-v2": "https://huggingface.co/albert/albert-base-v2/resolve/main/tokenizer.json",
        "albert/albert-large-v2": "https://huggingface.co/albert/albert-large-v2/resolve/main/tokenizer.json",
        "albert/albert-xlarge-v2": "https://huggingface.co/albert/albert-xlarge-v2/resolve/main/tokenizer.json",
        "albert/albert-xxlarge-v2": "https://huggingface.co/albert/albert-xxlarge-v2/resolve/main/tokenizer.json",
    },
}

# 定义一个空的类结束符号，用于关闭类的定义块

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

# 预训练模型位置编码嵌入大小的字典，包含不同ALBERT模型的名称及其对应的嵌入大小

SPIECE_UNDERLINE = "▁"

# 定义一个特殊符号，表示未分词的部分（用于分词模型的特殊符号）

class AlbertTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”ALBERT分词器（由HuggingFace的tokenizers库支持）。基于Unigram模型。
    该分词器继承自PreTrainedTokenizerFast类，包含大部分主要方法。用户可以参考这个超类获取更多关于这些方法的信息。
    """
    
    # 类的文档字符串，描述了这个类的作用和继承关系
    # 定义函数参数说明文档，用于指定 SentencePiece 文件，其中包含用于实例化分词器的词汇表
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 *.spm* 扩展名），包含实例化分词器所需的词汇表。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        remove_space (`bool`, *optional*, defaults to `True`):
            是否在分词时去除文本中的空格（去除字符串前后的多余空格）。
        keep_accents (`bool`, *optional*, defaults to `False`):
            是否在分词时保留重音符号。
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            在预训练期间用作序列开头的特殊标记。在构建使用特殊标记的序列时，实际上使用的标记是 `cls_token`。
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            序列结束的特殊标记。在构建使用特殊标记的序列时，实际使用的是 `sep_token`。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中没有的标记将无法转换为 ID，并被设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔标记，用于从多个序列构建一个序列，例如用于序列分类或文本与问题之间的问答。也用作构建带有特殊标记的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            填充标记，例如在对不同长度的序列进行批处理时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，在进行序列分类（整个序列而不是每个标记的分类）时使用。在使用特殊标记构建序列时，它是序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            掩码值标记。用于掩码语言建模的训练中，模型将尝试预测此标记。
    """

    # 从全局变量中获取常量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = AlbertTokenizer
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # 初始化函数，用于实例化对象并设置初始属性值
        # 设置 mask_token，使其表现得像一个普通单词，包括前面的空格，并且在原始文本中也包含，应该在非规范化的句子中匹配。
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 调用父类的初始化函数，传递相应的参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
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

        # 设置对象的属性值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存缓慢的分词器，基于是否存在 vocab_file 文件
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。
        ALBERT 序列的格式如下：

        - 单个序列： `[CLS] X [SEP]`
        - 序列对： `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对

        Returns:
            `List[int]`: 带有适当特殊标记的输入 ID 列表
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        # 根据输入的序列构建 token_type_ids，用于区分两个序列的标识
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define separator and class tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # Check if token_ids_1 is None; return mask with only the first portion
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return full mask including both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if the fast tokenizer can save vocabulary; raise error if not
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if save_directory exists; log error and return if not
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Copy vocabulary file if it's not already in the specified output path
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path of the saved vocabulary file
        return (out_vocab_file,)
```