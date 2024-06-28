# `.\models\vits\tokenization_vits.py`

```py
# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors, the MMS-TTS Authors and the HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for VITS."""


import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_phonemizer_available, logging


if is_phonemizer_available():
    import phonemizer


logger = logging.get_logger(__name__)

# 定义词汇表文件名字典常量
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

# 预训练模型的词汇表文件映射常量
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mms-tts-eng": "https://huggingface.co/facebook/mms-tts-eng/resolve/main/vocab.json",
    }
}

# 预训练模型的最大输入尺寸常量
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 该模型没有最大输入长度限制
    "facebook/mms-tts-eng": 4096,
}


def has_non_roman_characters(input_string):
    # 查找输入字符串中是否包含非罗马字符
    non_roman_pattern = re.compile(r"[^\x00-\x7F]")

    # 在输入字符串中搜索非罗马字符
    match = non_roman_pattern.search(input_string)
    has_non_roman = match is not None
    return has_non_roman


class VitsTokenizer(PreTrainedTokenizer):
    """
    Construct a VITS tokenizer. Also supports MMS-TTS.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        language (`str`, *optional*):
            Language identifier.
        add_blank (`bool`, *optional*, defaults to `True`):
            Whether to insert token id 0 in between the other tokens.
        normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the input text by removing all casing and punctuation.
        phonemize (`bool`, *optional*, defaults to `True`):
            Whether to convert the input text into phonemes.
        is_uroman (`bool`, *optional*, defaults to `False`):
            Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.
    """

    # 词汇表文件名字典常量
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇表文件映射常量
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸常量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_file,                # 词汇表文件的路径
        pad_token="<pad>",         # 填充标记，默认为"<pad>"
        unk_token="<unk>",         # 未知标记，默认为"<unk>"
        language=None,             # 语言设置，默认为None
        add_blank=True,            # 是否添加空白标记，默认为True
        normalize=True,            # 是否进行文本规范化，默认为True
        phonemize=True,            # 是否进行音素化，默认为True
        is_uroman=False,           # 是否为乌罗马尼亚语，默认为False
        **kwargs,                  # 其他关键字参数
    ) -> None:
        # 使用指定的编码打开词汇表文件，并加载为字典形式到self.encoder中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 根据self.encoder创建反向字典，用于从编码解码为原始词汇
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.language = language        # 设置对象的语言属性
        self.add_blank = add_blank      # 设置对象是否添加空白标记的属性
        self.normalize = normalize      # 设置对象是否进行文本规范化的属性
        self.phonemize = phonemize      # 设置对象是否进行音素化的属性
        self.is_uroman = is_uroman      # 设置对象是否为乌罗马尼亚语的属性

        # 调用父类的初始化方法，传递相同的参数和额外的关键字参数
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            language=language,
            add_blank=add_blank,
            normalize=normalize,
            phonemize=phonemize,
            is_uroman=is_uroman,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小，即self.encoder中条目的数量
        return len(self.encoder)

    def get_vocab(self):
        # 创建并返回词汇表的字典，将编码映射到词汇
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)  # 将额外添加的标记编码映射也加入到词汇表中
        return vocab

    def normalize_text(self, input_string):
        """Lowercase the input string, respecting any special token ids that may be part or entirely upper-cased."""
        # 获取所有词汇（包括特殊标记）的列表
        all_vocabulary = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())
        filtered_text = ""

        i = 0
        # 遍历输入字符串的每个字符
        while i < len(input_string):
            found_match = False
            # 遍历词汇表中的每个词汇
            for word in all_vocabulary:
                # 如果输入字符串中的当前位置开始的子串与词汇匹配
                if input_string[i : i + len(word)] == word:
                    filtered_text += word  # 将匹配的词汇添加到过滤后的文本中
                    i += len(word)          # 更新当前位置
                    found_match = True
                    break

            # 如果没有找到匹配的词汇，则将当前字符转换为小写添加到过滤后的文本中
            if not found_match:
                filtered_text += input_string[i].lower()
                i += 1

        return filtered_text

    def _preprocess_char(self, text):
        """Special treatment of characters in certain languages"""
        # 如果语言设置为罗马尼亚语（ron），则将特定字符进行替换处理
        if self.language == "ron":
            text = text.replace("ț", "ţ")  # 将字符"ț"替换为"ţ"
        return text

    def prepare_for_tokenization(
        self, text: str,                        # 输入的文本字符串
        is_split_into_words: bool = False,      # 是否已经分割为单词，默认为False
        normalize: Optional[bool] = None,       # 是否进行文本规范化，可选参数，默认为None
        **kwargs                               # 其他关键字参数
    ):
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize.
            normalize (`bool`, *optional*, defaults to `None`):
                Whether or not to apply punctuation and casing normalization to the text inputs. Typically, VITS is
                trained on lower-cased and un-punctuated text. Hence, normalization is used to ensure that the input
                text consists only of lower-case characters.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        # Determine the normalization setting to use
        normalize = normalize if normalize is not None else self.normalize

        # Normalize the text if required
        if normalize:
            # Normalize text casing and punctuation
            text = self.normalize_text(text)

        # Preprocess text to filter unwanted characters
        filtered_text = self._preprocess_char(text)

        # Check for non-Roman characters if the tokenizer is set to uroman
        if has_non_roman_characters(filtered_text) and self.is_uroman:
            # Issue a warning if non-Roman characters are detected
            logger.warning(
                "Text to the tokenizer contains non-Roman characters. Ensure the `uroman` Romanizer is "
                "applied to the text prior to passing it to the tokenizer. See "
                "`https://github.com/isi-nlp/uroman` for details."
            )

        # Apply phonemization if enabled
        if self.phonemize:
            # Check if phonemizer is available
            if not is_phonemizer_available():
                # Raise an error if phonemizer is not installed
                raise ImportError("Please install the `phonemizer` Python package to use this tokenizer.")

            # Phonemize the filtered text
            filtered_text = phonemizer.phonemize(
                filtered_text,
                language="en-us",
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
            )
            # Replace multiple spaces with a single space
            filtered_text = re.sub(r"\s+", " ", filtered_text)
        elif normalize:
            # Strip characters outside of the vocabulary (punctuation)
            filtered_text = "".join(list(filter(lambda char: char in self.encoder, filtered_text))).strip()

        # Return the processed text and remaining kwargs
        return filtered_text, kwargs
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string by inserting the `<pad>` token at the boundary between adjacent characters."""
        # 将字符串按字符分割为列表
        tokens = list(text)

        # 如果设置了添加空白标记，将空白标记插入相邻字符之间
        if self.add_blank:
            # 创建一个新列表，用于在字符之间插入空白标记
            interspersed = [self._convert_id_to_token(0)] * (len(tokens) * 2 + 1)
            interspersed[1::2] = tokens  # 将原始字符插入新列表的奇数位置
            tokens = interspersed

        return tokens  # 返回处理后的标记列表

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 如果设置了添加空白标记并且标记列表长度大于1，只返回奇数位置的标记组成的字符串
        if self.add_blank and len(tokens) > 1:
            tokens = tokens[1::2]
        return "".join(tokens)  # 将标记列表连接为字符串并返回

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将标记转换为对应的 ID，如果标记不存在则使用未知标记的 ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据索引从词汇表中获取对应的标记
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Union[Tuple[str], None]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回空值
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 拼接词汇表文件的完整路径和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 将词汇表以 JSON 格式写入到文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)  # 返回保存的词汇表文件路径元组
```