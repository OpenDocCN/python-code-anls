# `.\models\mgp_str\tokenization_mgp_str.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Tokenization classes for MGT-STR CHAR.
"""

import json  # 导入json模块，用于处理JSON格式的数据
import os    # 导入os模块，提供了与操作系统交互的功能
from typing import Optional, Tuple   # 导入类型提示相关的模块

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器的基类
from ...utils import logging   # 导入日志记录模块

logger = logging.get_logger(__name__)   # 获取当前模块的日志记录器对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}   # 定义词汇表文件名映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "mgp-str": "https://huggingface.co/alibaba-damo/mgp-str-base/blob/main/vocab.json",
    }
}   # 预训练词汇文件映射，指定了不同预训练模型的词汇文件路径

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mgp-str": 27}   # 预训练位置嵌入的尺寸映射

class MgpstrTokenizer(PreTrainedTokenizer):
    """
    Construct a MGP-STR char tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"[GO]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"[GO]"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"[s]"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"[GO]"`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
    """

    vocab_files_names = VOCAB_FILES_NAMES   # 设置词汇文件名映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP   # 设置预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES   # 设置预训练位置嵌入的尺寸

    def __init__(self, vocab_file, unk_token="[GO]", bos_token="[GO]", eos_token="[s]", pad_token="[GO]", **kwargs):
        """
        Initialize a tokenizer instance.

        Args:
            vocab_file (`str`):
                Path to the vocabulary file.
            unk_token (`str`, *optional*, defaults to `"[GO]"`):
                The unknown token.
            bos_token (`str`, *optional*, defaults to `"[GO]"`):
                The beginning of sequence token.
            eos_token (`str`, *optional*, defaults to `"[s]"`):
                The end of sequence token.
            pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"[GO]"`):
                The padding token used in batching.
            **kwargs:
                Additional keyword arguments passed to the parent class constructor.
        """
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)   # 从指定路径加载词汇表文件，并转换为字典形式
        self.decoder = {v: k for k, v in self.vocab.items()}   # 创建反向词汇表，用于将ID转换为对应的词汇
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """
        Return the size of the vocabulary.

        Returns:
            int: Number of tokens in the vocabulary.
        """
        return len(self.vocab)   # 返回词汇表中词汇的数量

    def get_vocab(self):
        """
        Get the vocabulary (including any additional tokens).

        Returns:
            dict: A dictionary containing the vocabulary tokens and their IDs.
        """
        vocab = dict(self.vocab).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab   # 返回包含额外token的完整词汇表字典
    # 将文本字符串进行分词处理，返回字符级别的标记列表
    def _tokenize(self, text):
        char_tokens = []  # 初始化一个空列表，用于存储字符级别的标记
        for s in text:
            char_tokens.extend(s)  # 将每个字符作为一个标记加入到列表中
        return char_tokens  # 返回字符级别的标记列表

    # 根据词汇表将标记转换为对应的 ID
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))  # 返回标记对应的 ID，如果标记不存在则使用未知标记的 ID

    # 根据词汇表将 ID 转换为对应的标记
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)  # 返回给定 ID 对应的标记

    # 将词汇表保存到指定的目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):  # 检查保存目录是否存在，如果不存在则记录错误并返回
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return  # 返回空值，表示保存操作未成功

        # 构建词汇表文件的路径，文件名根据可选的前缀和预定义的文件名组成
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 将词汇表以 JSON 格式写入到文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)  # 返回保存的词汇表文件路径的元组
```