# `.\models\fnet\tokenization_fnet_fast.py`

```
# coding=utf-8
# Copyright 2021 Google AI, Google Brain and the HuggingFace Inc. team.
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
""" Tokenization classes for FNet model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# Check if sentencepiece library is available
if is_sentencepiece_available():
    # Import the specific tokenizer for FNet from local module
    from .tokenization_fnet import FNetTokenizer
else:
    # Set FNetTokenizer to None if sentencepiece is not available
    FNetTokenizer = None

# Initialize logger for this module
logger = logging.get_logger(__name__)

# Define vocabulary files names expected by the tokenizer
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# Define pretrained vocab files mapping for different model configurations
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/tokenizer.json",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/tokenizer.json",
    },
}

# Define sizes of positional embeddings for different model configurations
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}

# Special token used by sentencepiece for word beginning pieces
SPIECE_UNDERLINE = "▁"


class FNetTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" FNetTokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`AlbertTokenizerFast`]. Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods
    """
    # 声明全局变量，包含预定义的词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 包含预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 包含预训练位置嵌入的最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表，用于初始化
    model_input_names = ["input_ids", "token_type_ids"]
    # 慢速分词器的类别，使用了 FNetTokenizer
    slow_tokenizer_class = FNetTokenizer

    # 初始化方法，接受多个可选参数来配置分词器的行为
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
        ):
            # 如果 mask_token 是字符串，则创建一个 AddedToken 对象，保留前导空格但不保留后导空格
            mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
            # 如果 cls_token 是字符串，则创建一个 AddedToken 对象，不保留前导和后导空格
            cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
            # 如果 sep_token 是字符串，则创建一个 AddedToken 对象，不保留前导和后导空格
            sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
    
            # 调用父类的初始化方法，设置基本的 tokenizer 参数
            super().__init__(
                vocab_file,
                tokenizer_file=tokenizer_file,
                do_lower_case=do_lower_case,
                remove_space=remove_space,
                keep_accents=keep_accents,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                **kwargs,
            )
    
            # 设置当前对象的属性值
            self.do_lower_case = do_lower_case
            self.remove_space = remove_space
            self.keep_accents = keep_accents
            self.vocab_file = vocab_file
    
        @property
        def can_save_slow_tokenizer(self) -> bool:
            # 检查词汇文件是否存在，从而判断是否可以保存慢速 tokenizer
            return os.path.isfile(self.vocab_file) if self.vocab_file else False
    
        def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
            """
            通过连接并添加特殊 token 构建用于序列分类任务的模型输入。FNet 序列有以下格式：

            - 单序列：`[CLS] X [SEP]`
            - 序列对：`[CLS] A [SEP] B [SEP]`

            Args:
                token_ids_0 (`List[int]`):
                    要添加特殊 token 的 ID 列表
                token_ids_1 (`List[int]`, *optional*):
                    第二个序列的可选 ID 列表，用于序列对任务

            Returns:
                `List[int]`: 包含适当特殊 token 的输入 ID 列表
            """
            sep = [self.sep_token_id]
            cls = [self.cls_token_id]
            if token_ids_1 is None:
                return cls + token_ids_0 + sep
            return cls + token_ids_0 + sep + token_ids_1 + sep
    
        def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An FNet
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
        # Define the special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # Check if token_ids_1 is None; if so, return a mask for only the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Otherwise, concatenate masks for both sequences (first sequence: 0s, second sequence: 1s)
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Ensure the save_directory exists; if not, log an error and return None
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocab_file path is different from the desired out_vocab_file path, copy the vocab_file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```