# `.\models\esm\tokenization_esm.py`

```py
# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ESM."""
import os
from typing import List, Optional

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取名为 logging 的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型对应的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/esm2_t6_8M_UR50D": "https://huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/vocab.txt",
        "facebook/esm2_t12_35M_UR50D": "https://huggingface.co/facebook/esm2_t12_35M_UR50D/resolve/main/vocab.txt",
    },
}

# 定义预训练模型对应的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 1024,
    "facebook/esm2_t12_35M_UR50D": 1024,
}


def load_vocab_file(vocab_file):
    # 打开给定路径的词汇文件，并将内容按行读取为列表
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]


class EsmTokenizer(PreTrainedTokenizer):
    """
    Constructs an ESM tokenizer.
    """

    # 设置类属性：词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置类属性：预训练模型对应的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设置类属性：预训练模型对应的位置嵌入大小映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置类属性：模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        **kwargs,
    ):
        # 加载词汇文件中的所有词汇，并构建词汇表
        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            **kwargs,
        )

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        # 将所有词汇加入到不需要拆分的特殊标记列表中
        self.unique_no_split_tokens = self.all_tokens
        # 更新基于特殊标记列表的 Trie 数据结构
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        # 根据索引将其转换为对应的词汇，若索引不存在则返回未知标记
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        # 根据词汇将其转换为对应的索引，若词汇不存在则返回未知标记的索引
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))
    # 将输入文本按空格分割，返回分割后的列表作为结果
    def _tokenize(self, text, **kwargs):
        return text.split()

    # 返回包含基础词汇的字典，包括_token_to_id和added_tokens_encoder的合并
    def get_vocab(self):
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    # 根据给定的token返回其对应的id，如果token不存在则返回unk_token对应的id
    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    # 根据给定的index返回对应的token，如果index不存在则返回unk_token
    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    # 构建包含特殊token的输入列表，处理单个或两个序列的情况
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id]  # ESM词汇表中没有sep token
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + token_ids_1 + sep  # 多个输入始终有一个EOS token

    # 获取不包含特殊token的token列表的特殊token掩码
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        检索没有添加特殊token的token列表的序列id。当使用tokenizer的`prepare_for_model`或`encode_plus`方法添加特殊token时调用此方法。
        
        Args:
            token_ids_0 (`List[int]`):
                第一个序列的id列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的id列表。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                token列表是否已经格式化包含了模型的特殊token。

        Returns:
            一个整数列表，范围为[0, 1]：1表示特殊token，0表示序列token。
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        
        # 创建一个mask列表，标识特殊token的位置
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    # 将词汇表保存到指定目录下的文件中，文件名由filename_prefix和vocab.txt组成
    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    # 返回词汇表的大小，即all_tokens的长度
    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)
```