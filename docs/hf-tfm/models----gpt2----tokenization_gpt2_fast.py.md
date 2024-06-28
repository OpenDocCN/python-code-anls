# `.\models\gpt2\tokenization_gpt2_fast.py`

```py
# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""

# 导入必要的模块
import json  # 导入处理 JSON 格式的模块
from typing import Optional, Tuple  # 导入类型提示相关的模块

from tokenizers import pre_tokenizers  # 从 tokenizers 包中导入预处理器相关功能

from ...tokenization_utils_base import BatchEncoding  # 导入批量编码类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入快速预训练分词器类
from ...utils import logging  # 导入日志记录工具
from .tokenization_gpt2 import GPT2Tokenizer  # 从当前目录中导入 GPT2Tokenizer 类

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义用于存储文件名的常量字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/vocab.json",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/vocab.json",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/vocab.json",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/merges.txt",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/merges.txt",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/merges.txt",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/tokenizer.json",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/tokenizer.json",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/tokenizer.json",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/tokenizer.json",
    },
}

# 预训练位置嵌入大小映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 创建一个字典，包含不同的模型名称作为键，每个模型的默认大小（1024）作为值
    "openai-community/gpt2": 1024,
    "openai-community/gpt2-medium": 1024,
    "openai-community/gpt2-large": 1024,
    "openai-community/gpt2-xl": 1024,
    "distilbert/distilgpt2": 1024,
}

class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import GPT2TokenizerFast

    >>> tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

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
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = GPT2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        """
        Initialize the GPT2TokenizerFast class.

        Args:
            vocab_file (`str`, *optional*):
                Path to the vocabulary file.
            merges_file (`str`, *optional*):
                Path to the merges file.
            tokenizer_file (`str`, *optional*):
                Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
                contains everything needed to load the tokenizer.
            unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
                The unknown token.
            bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
                The beginning of sequence token.
            eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
                The end of sequence token.
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                Whether or not to add an initial space to the input. This allows to treat the leading word just as any
                other word. (GPT2 tokenizer detect beginning of words by the preceding space).
            **kwargs:
                Additional arguments passed to the superclass.
        """
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.add_bos_token = kwargs.pop("add_bos_token", False)

        # 获取当前的预处理器状态
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 检查预处理器是否需要添加前缀空格，如果需要则更新其状态
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 检查是否已经将输入分割为单词
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果需要使用预分词的输入，确保实例化时设置了 add_prefix_space=True
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 检查是否已经将输入分割为单词
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果需要使用预分词的输入，确保实例化时设置了 add_prefix_space=True
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存词汇表到指定的目录，返回保存的文件名
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template 复制而来
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 若没有为这个分词器定义聊天模板，则使用默认的模板，并记录警告信息
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```