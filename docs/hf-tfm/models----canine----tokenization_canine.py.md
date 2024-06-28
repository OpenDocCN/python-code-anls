# `.\models\canine\tokenization_canine.py`

```
# coding=utf-8
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
Tokenization classes for CANINE.
"""

from typing import Dict, List, Optional

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型位置嵌入的大小，这里仅有一个预训练模型 'nielsr/canine-s'，其位置嵌入大小为 2048
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "nielsr/canine-s": 2048,
}

# Unicode 定义了总共 1,114,112 个“代码点”
UNICODE_VOCAB_SIZE = 1114112

# 下面是定义特殊伪字符的规范代码点的常量
# 从 https://github.com/google-research/language/blob/master/language/canine/special_codepoints.py 复制而来
PAD = 0           # 填充字符
CLS = 0xE000      # 序列的开始标记
SEP = 0xE001      # 序列的分隔符
BOS = 0xE002      # 句子的开始标记
MASK = 0xE003     # 掩码标记
RESERVED = 0xE004 # 保留标记

# 将特殊代码点映射到人类可读的名称
SPECIAL_CODEPOINTS: Dict[int, str] = {
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

# 将特殊代码点的人类可读名称映射回其代码点值
SPECIAL_CODEPOINTS_BY_NAME: Dict[str, int] = {name: codepoint for codepoint, name in SPECIAL_CODEPOINTS.items()}


class CanineTokenizer(PreTrainedTokenizer):
    """
    构建 CANINE 分词器（即字符分割器）。它将文本转换为字符序列，然后将每个字符转换为其 Unicode 代码点。

    [`CanineTokenizer`] 继承自 [`PreTrainedTokenizer`]。

    有关参数使用示例和文档，请参阅超类 [`PreTrainedTokenizer`]。

    Args:
        model_max_length (`int`, *optional*, 默认为 2048):
                模型接受的最大句子长度。
    """

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        bos_token=chr(CLS),
        eos_token=chr(SEP),
        sep_token=chr(SEP),
        cls_token=chr(CLS),
        pad_token=chr(PAD),
        mask_token=chr(MASK),
        add_prefix_space=False,
        model_max_length=2048,
        **kwargs,
    ):
        # 如果提供的特殊符号是字符串，则将其封装为 AddedToken 对象，确保左右两侧的空格不会被去除
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 如果提供的 mask_token 是字符串，则创建 AddedToken 对象，并确保去除左侧空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 创建一个字典，用于查找特殊符号的 ID
        self._special_codepoints: Dict[str, int] = {}
        for codepoint, name in SPECIAL_CODEPOINTS.items():
            self._special_codepoints[name] = codepoint

        # 创建一个字典，用于查找特殊符号 ID 对应的字符串形式
        self._special_codepoint_strings: Dict[int, str] = {
            codepoint: name for name, codepoint in self._special_codepoints.items()
        }

        # 设置 Unicode 词汇表大小
        self._unicode_vocab_size = UNICODE_VOCAB_SIZE
        # 计算特殊符号的数量
        self._num_special_tokens = len(self._special_codepoints)

        # 调用父类的构造函数，初始化对象
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        # 返回 Unicode 词汇表的大小
        return self._unicode_vocab_size

    def get_vocab(self):
        # 创建并返回一个词汇表，包括所有 Unicode 字符和额外添加的 tokens
        vocab = {chr(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string (i.e. perform character splitting)."""
        # 将字符串拆分为单个字符，并返回列表形式
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (i.e. a Unicode character) in an id (i.e. its integer Unicode code point value)."""
        try:
            # 将 Unicode 字符转换为其整数 Unicode 码点值
            return ord(token)
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a Unicode code point (integer) in a token (str). In case it's a special code point, convert to
        human-readable format.
        """
        try:
            # 如果索引是特殊代码点，则转换为人类可读的格式
            if index in SPECIAL_CODEPOINTS:
                return SPECIAL_CODEPOINTS[index]
            # 否则，将整数转换为 Unicode 字符
            return chr(index)
        except TypeError:
            raise ValueError(f"invalid id: {index}")

    def convert_tokens_to_string(self, tokens):
        # 将 token 列表连接成一个字符串并返回
        return "".join(tokens)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A CANINE sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # Define special tokens
        sep = [self.sep_token_id]  # SEP token ID
        cls = [self.cls_token_id]  # CLS token ID

        # Construct input with special tokens
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # Check if special tokens are already present
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Initialize mask with special tokens
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from token list sequences.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs indicating the type of each token in the input sequences.
        """
        # Initialize token type IDs
        token_type_ids = [0] * len(token_ids_0)

        # If token_ids_1 is provided, set its token type IDs to 1
        if token_ids_1 is not None:
            token_type_ids += [1] * len(token_ids_1)

        return token_type_ids
    # 定义函数，用于生成用于序列对分类任务的掩码。CANINE序列对掩码的格式如下：
    #
    # ```
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | 第一个序列       | 第二个序列       |
    # ```
    #
    # 如果 `token_ids_1` 是 `None`，则此方法只返回掩码的第一个部分（全为0）。
    #
    # Args:
    #     token_ids_0 (`List[int]`):
    #         第一个序列的ID列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         第二个序列的ID列表，用于序列对。
    #
    # Returns:
    #     `List[int]`: 根据给定序列(s)生成的 [token type IDs](../glossary#token-type-ids) 列表。
    def create_sequence_pair_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]  # 分隔符的ID列表
        cls = [self.cls_token_id]  # 类别开始的ID列表

        result = len(cls + token_ids_0 + sep) * [0]  # 初始化结果为第一个序列部分全为0的掩码
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]  # 如果存在第二个序列，则将其加入到掩码中，第二部分全为1
        return result  # 返回生成的掩码列表

    # CanineTokenizer没有词汇文件
    # 定义函数，用于保存词汇表（空操作）
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()  # 返回空元组，表示保存操作无需实际执行
```