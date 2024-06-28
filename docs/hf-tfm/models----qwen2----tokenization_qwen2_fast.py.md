# `.\models\qwen2\tokenization_qwen2_fast.py`

```
# coding=utf-8
# 版权所有 2024 年 Qwen 团队、阿里巴巴集团和 HuggingFace 公司。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”基础分发，
# 没有任何明示或暗示的保证或条件。
# 有关更多信息，请参阅许可证。

"""Qwen2 的标记化类。"""

from typing import Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_qwen2 import Qwen2Tokenizer

logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# 定义预训练模型所需的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/merges.txt"},
    "tokenizer_file": {
        "qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/tokenizer.json"
    },
}

# 定义模型的最大输入尺寸
MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”的 Qwen2 分词器（基于 HuggingFace 的 *tokenizers* 库）。基于字节级的 Byte-Pair-Encoding。

    与 GPT2Tokenizer 类似，此分词器经过训练，将空格视为标记的一部分，因此一个单词在句子开头（没有空格）和其他位置将被编码为不同的标记：

    ```python
    >>> from transformers import Qwen2TokenizerFast

    >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    这是预期的行为。

    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional`):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    # These variables define certain constants for the tokenizer configuration
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        """
        Initializes a new instance of the Qwen2Tokenizer class.
        
        Args:
            vocab_file (str, optional): Path to the vocabulary file.
            merges_file (str, optional): Path to the merges file.
            tokenizer_file (str, optional): Path to tokenizers file.
            unk_token (str, optional, default="<|endoftext|>"): The unknown token.
            bos_token (str, optional): The beginning of sequence token.
            eos_token (str, optional, default="<|endoftext|>"): The end of sequence token.
            pad_token (str, optional, default="<|endoftext|>"): The padding token.
            **kwargs: Additional keyword arguments passed to the base class constructor.
        """
        
        # Set bos_token, eos_token, unk_token, and pad_token as AddedToken objects if they are strings
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )
        
        # Call the base class constructor with the provided arguments
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
    # 从 transformers 库中 GPT2TokenizerFast 类的 save_vocabulary 方法复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用内部的 tokenizer 模块的 save 方法，将模型保存到指定的目录中，并使用给定的前缀作为文件名前缀
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名组成的元组
        return tuple(files)
```