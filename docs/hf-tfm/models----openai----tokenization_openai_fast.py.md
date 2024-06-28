# `.\models\openai\tokenization_openai_fast.py`

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
"""Fast Tokenization classes for OpenAI GPT."""

from typing import Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入基于 tokenizers 库的预训练标记器
from ...utils import logging  # 导入日志模块
from .tokenization_openai import OpenAIGPTTokenizer  # 导入 OpenAIGPTTokenizer 类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/vocab.json"
    },
    "merges_file": {
        "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/merges.txt"
    },
    "tokenizer_file": {
        "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/tokenizer.json"
    },
}

# 预训练模型的位置嵌入尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-community/openai-gpt": 512,
}

# OpenAIGPTTokenizerFast 类，继承自 PreTrainedTokenizerFast
class OpenAIGPTTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT Tokenizer (backed by HuggingFace's *tokenizers* library). Based on Byte-Pair-Encoding with
    the following peculiarities:

    - lower case all inputs
    - uses BERT's BasicTokenizer for pre-BPE tokenization

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件的名称映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练模型的词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置预训练模型的最大输入尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称列表
    slow_tokenizer_class = OpenAIGPTTokenizer  # 慢速标记器的类

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token="<unk>", **kwargs):
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, **kwargs)
        # 初始化函数，调用父类的构造函数，并传递参数

    @property
    def do_lower_case(self):
        return True  # 返回是否将输入文本转换为小写的标志
    # 定义一个方法用于保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用内部的分词器模型的保存方法，将词汇表保存到指定目录下，使用可选的文件名前缀
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 将返回的文件名转换成元组并返回
        return tuple(files)
```