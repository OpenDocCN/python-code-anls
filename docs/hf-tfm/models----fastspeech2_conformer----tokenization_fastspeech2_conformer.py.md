# `.\models\fastspeech2_conformer\tokenization_fastspeech2_conformer.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Tokenization classes for FastSpeech2Conformer.
"""
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
from typing import Optional, Tuple  # 导入类型提示相关模块

import regex  # 导入正则表达式模块

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器基类
from ...utils import logging, requires_backends  # 导入日志和后端依赖的模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义词汇文件名映射字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

# 定义预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "espnet/fastspeech2_conformer": "https://huggingface.co/espnet/fastspeech2_conformer/raw/main/vocab.json",
    },
}

# 定义预训练模型的位置编码尺寸映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 设置为相对较大的任意数字，因为模型输入不受相对位置编码的限制
    "espnet/fastspeech2_conformer": 4096,
}


class FastSpeech2ConformerTokenizer(PreTrainedTokenizer):
    """
    Construct a FastSpeech2Conformer tokenizer.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The begin of sequence token. Note that for FastSpeech2, it is the same as the `eos_token`.
        eos_token (`str`, *optional*, defaults to `"<sos/eos>"`):
            The end of sequence token. Note that for FastSpeech2, it is the same as the `bos_token`.
        pad_token (`str`, *optional*, defaults to `"<blank>"`):
            The token used for padding, for example when batching sequences of different lengths.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        should_strip_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the spaces from the list of tokens.
    """

    # 设置词汇文件名映射
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设置模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 设置预训练位置编码的最大尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        bos_token="<sos/eos>",
        eos_token="<sos/eos>",
        pad_token="<blank>",
        unk_token="<unk>",
        should_strip_spaces=False,
        **kwargs,
    ):
        # 检查是否需要引入"g2p_en"后端
        requires_backends(self, "g2p_en")

        # 使用 UTF-8 编码打开词汇文件，并加载为 JSON 格式的编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 导入 g2p_en 库并初始化 g2p 对象
        import g2p_en
        self.g2p = g2p_en.G2p()

        # 创建反向的编码器解码器映射关系
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 调用父类的初始化方法，设置特殊标记和其他参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            should_strip_spaces=should_strip_spaces,
            **kwargs,
        )

        # 设置是否应该去除空格的标志
        self.should_strip_spaces = should_strip_spaces

    @property
    def vocab_size(self):
        # 返回词汇表大小，即解码器的长度
        return len(self.decoder)

    def get_vocab(self):
        "Returns vocab as a dict"
        # 返回编码器和添加的特殊标记编码器的组合词汇表
        return dict(self.encoder, **self.added_tokens_encoder)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 扩展特殊符号
        text = regex.sub(";", ",", text)
        text = regex.sub(":", ",", text)
        text = regex.sub("-", " ", text)
        text = regex.sub("&", "and", text)

        # 去除不必要的符号
        text = regex.sub(r"[\(\)\[\]\<\>\"]+", "", text)

        # 去除空白字符
        text = regex.sub(r"\s+", " ", text)

        # 将文本转换为大写
        text = text.upper()

        return text, kwargs

    def _tokenize(self, text):
        """Returns a tokenized string."""
        # 使用 g2p 对文本进行音素化
        tokens = self.g2p(text)

        # 如果需要去除空格，则过滤掉空格字符
        if self.should_strip_spaces:
            tokens = list(filter(lambda s: s != " ", tokens))

        # 添加结束标记到 tokens
        tokens.append(self.eos_token)

        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用编码器将 token 转换为 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用解码器将 id 转换为 token
        return self.decoder.get(index, self.unk_token)

    # 重写因为音素无法可靠地转换回字符串
    def decode(self, token_ids, **kwargs):
        # 发出警告，由于一对多映射，音素不能可靠地转换为字符串，改为返回 token
        logger.warn(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, converting to tokens instead."
        )
        return self.convert_ids_to_tokens(token_ids)

    # 重写因为音素无法可靠地转换回字符串
    def convert_tokens_to_string(self, tokens, **kwargs):
        # 发出警告，由于一对多映射，音素不能可靠地转换为字符串，返回 tokens
        logger.warn(
            "Phonemes cannot be reliably converted to a string due to the one-many mapping, returning the tokens."
        )
        return tokens
    # 将词汇表和特殊标记文件保存到指定目录中。

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建词汇表文件的路径，包括可选的文件名前缀和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 打开词汇表文件，并将词汇表内容以 UTF-8 编码写入文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False))

        # 返回保存的文件路径的元组
        return (vocab_file,)

    # 返回对象的状态信息，用于序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        # 移除 g2p 对象，以便对象可以被序列化
        state["g2p"] = None
        return state

    # 设置对象的状态，用于反序列化
    def __setstate__(self, d):
        self.__dict__ = d

        try:
            # 尝试导入 g2p_en 库，并初始化 g2p 对象
            import g2p_en

            self.g2p = g2p_en.G2p()
        except ImportError:
            # 如果导入失败，抛出 ImportError，并提供安装 g2p-en 库的链接
            raise ImportError(
                "You need to install g2p-en to use FastSpeech2ConformerTokenizer. "
                "See https://pypi.org/project/g2p-en/ for installation."
            )
```