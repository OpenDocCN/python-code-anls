# `.\models\clip\tokenization_clip_fast.py`

```
# coding=utf-8
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""
Tokenization classes for OpenAI GPT.
"""

from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers  # 导入 tokenizers 库中的 pre_tokenizers 模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入快速分词器的基类
from ...utils import logging  # 导入日志工具
from .tokenization_clip import CLIPTokenizer  # 导入 CLIPTokenizer 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai/clip-vit-base-patch32": (
            "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/clip-vit-base-patch32": 77,
}


class CLIPTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CLIP tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件的名称字典
    # 使用预先定义的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 使用预先定义的模型输入最大尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速分词器的类别为 CLIPTokenizer
    slow_tokenizer_class = CLIPTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # hack to enable padding
        **kwargs,
    ):
        # 调用父类的初始化方法，设置词汇、合并文件及分词器文件
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        # 检查后端分词器的预处理器是否为序列预处理器，否则抛出值错误异常
        if not isinstance(self.backend_tokenizer.pre_tokenizer, pre_tokenizers.Sequence):
            raise ValueError(
                "The `backend_tokenizer` provided does not match the expected format. The CLIP tokenizer has been"
                " heavily modified from transformers version 4.17.0. You need to convert the tokenizer you are using"
                " to be compatible with this version.The easiest way to do so is"
                ' `CLIPTokenizerFast.from_pretrained("path_to_local_folder_or_hub_repo, from_slow=True)`. If you want'
                " to use your existing tokenizer, you will have to revert to a version prior to 4.17.0 of"
                " transformers."
            )

        # 修改后端分词器的解码方法，通过添加空格以确保正确的解码
        self._wrap_decode_method_backend_tokenizer()

    # 非常丑陋的hack，以使填充能够正确解码，详细见 https://github.com/huggingface/tokenizers/issues/872
    def _wrap_decode_method_backend_tokenizer(self):
        # 保存原始的解码方法
        orig_decode_method = self.backend_tokenizer.decode

        # 定义新的解码方法，替换结束词后缀为空格并去除两侧空格
        def new_decode_method(*args, **kwargs):
            text = orig_decode_method(*args, **kwargs)
            text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, " ").strip()
            return text

        # 覆盖后端分词器的解码方法为新定义的方法
        self.backend_tokenizer.decode = new_decode_method

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 以下代码行需要继续添加注释
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 定义起始和结束特殊标记的 ID 列表
        bos_token = [self.bos_token_id]
        eos_token = [self.eos_token_id]

        # 如果只有一个序列（单个文本），返回带有特殊标记的输入 ID 列表
        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]
        
        # 如果有两个序列（文本对），返回带有特殊标记的输入 ID 列表
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to the specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved files.

        Returns:
            `Tuple[str]`: Tuple containing the filenames of the saved vocabulary files.
        """
        # 调用内部的模型保存方法来保存词汇表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        
        # 返回保存的文件名组成的元组
        return tuple(files)
```