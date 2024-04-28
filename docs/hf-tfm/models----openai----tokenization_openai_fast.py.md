# `.\transformers\models\openai\tokenization_openai_fast.py`

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

# 导入相关模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_openai import OpenAIGPTTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表文件名和对应的文件
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型使用的词汇表文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
    "tokenizer_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/tokenizer.json"},
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-gpt": 512,
}

# OpenAI GPT 快速分词器类
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
    
    # 词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型使用的词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类
    slow_tokenizer_class = OpenAIGPTTokenizer
    
    # 初始化方法
    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token="<unk>", **kwargs):
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, **kwargs)
    
    # 返回是否进行小写处理
    @property
    def do_lower_case(self):
        return True
    # 定义保存词汇表的方法，接受保存目录和可选的文件名前缀作为参数，返回保存的文件名组成的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 Tokenizer 模型的 save 方法保存词汇表文件，返回保存的文件名列表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 将文件名列表转换为元组并返回
        return tuple(files)
```