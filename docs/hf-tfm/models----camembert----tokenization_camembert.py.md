# `.\models\camembert\tokenization_camembert.py`

```py
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License
""" Tokenization classes for Camembert model."""


import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入句子分词工具

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}  # 词汇文件名映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "almanach/camembert-base": "https://huggingface.co/almanach/camembert-base/resolve/main/sentencepiece.bpe.model",
    }
}  # 预训练模型词汇文件映射

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "almanach/camembert-base": 512,
}  # 预训练模型的位置嵌入大小映射

SPIECE_UNDERLINE = "▁"  # SentencePiece 分词中的特殊标记

class CamembertTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Construct a CamemBERT tokenizer. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名属性
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练模型词汇文件映射属性
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置预训练模型输入大小属性
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入名称列表

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 定义方法签名，无返回值
        # Mask token 表现得像普通单词，即包括其前面的空格
        mask_token = (
            # 如果 mask_token 是字符串，创建一个带有特殊属性的 AddedToken 对象
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 如果 sp_model_kwargs 为 None，则设为空字典，否则使用传入的 sp_model_kwargs
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 创建 SentencePieceProcessor 对象并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # HACK: 作者添加这些 token 有些隐晦的原因，因为它们已经在 sentencepiece 词汇中了
        # 对于 <s>、</s> 和 <unk>，建议手动设置这些 token
        self._added_tokens_decoder = {
            0: AddedToken("<s>NOTUSED", special=True),
            1: AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token,
            2: AddedToken("</s>NOTUSED", special=True),
            3: AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token,
            4: AddedToken("<unk>NOTUSED", special=True),
        }

        # fairseq 偏移量为 4，因为新增了 3 个 token，但偏移从 4 开始
        self.fairseq_offset = 4

        # legacy: camemebert 是一个特殊情况，需要确保 `"<unk>NOTUSED"` 在这里
        if "added_tokens_decoder" in kwargs:
            # 这是唯一一个需要这样做的类......
            # 原因是快速版本有一个完整的...
            kwargs["added_tokens_decoder"].update(self._added_tokens_decoder)

        # 调用父类的初始化方法，传递各种 token 和参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 词汇表大小是 self.sp_model 的长度，但添加的 token 在开头，所以加上 fairseq 偏移量
        return len(self.sp_model)

    def get_vocab(self):
        # 创建词汇表字典，包括已添加的 token 编码
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.fairseq_offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # 使用 SentencePiece 对文本进行编码成字符串列表
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """将 token (str) 转换为对应的 id，使用词汇表."""
        # 对于 camembert 特定的情况，3 和 4 都指向 unk token
        if self.sp_model.PieceToId(token) == 0:
            # 将 sentence piece unk token 转换为 fairseq unk token 的索引
            return self.unk_token_id
        return self.fairseq_offset + self.sp_model.PieceToId(token)
    # 使用索引值转换为对应的 token 字符串，通过 vocab 进行映射
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将一系列 token 字符串转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # TODO decode outputs do not match between fast and slow
        current_sub_tokens = []  # 用于存储当前正在构建的子 token 序列
        out_string = ""  # 最终合并的字符串结果
        prev_is_special = False  # 前一个 token 是否是特殊 token
        for token in tokens:
            # 检查当前 token 是否为特殊 token，如果是，则需要处理拼接
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "  # 如果前一个不是特殊 token，则添加空格分隔
                out_string += self.sp_model.decode(current_sub_tokens) + token  # 解码当前子 token 序列并拼接特殊 token
                prev_is_special = True  # 更新前一个 token 是特殊 token
                current_sub_tokens = []  # 重置子 token 序列
            else:
                current_sub_tokens.append(token)  # 添加当前 token 到子 token 序列
                prev_is_special = False  # 更新前一个 token 不是特殊 token
        out_string += self.sp_model.decode(current_sub_tokens)  # 处理剩余的子 token 序列并添加到最终结果中
        return out_string.strip()  # 返回去除首尾空格的字符串

    # 序列化对象状态以便进行存储
    def __getstate__(self):
        state = self.__dict__.copy()  # 创建对象状态的深拷贝副本
        state["sp_model"] = None  # 设置 sp_model 为 None，因为它无法被序列化
        return state

    # 反序列化对象状态并重新构建对象
    def __setstate__(self, d):
        self.__dict__ = d  # 恢复对象状态

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}  # 如果对象中不存在 sp_model_kwargs 属性，则创建空字典

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)  # 使用保存的参数重新创建 sp_model
        self.sp_model.Load(self.vocab_file)  # 加载之前保存的 vocab_file 文件

    # 将词汇表保存到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return  # 如果保存目录不存在，则记录错误并返回

        # 构建输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前 vocab_file 与输出路径不同且 vocab_file 是一个文件，则复制文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            # 如果 vocab_file 不存在，则将序列化后的 sp_model 写入到输出路径
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)  # 返回保存的词汇表文件路径的元组

    # 构建带有特殊 token 的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或者一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 CamemBERT 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个可选的序列对 ID 列表。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 返回带有特殊标记的单个序列输入 ID 列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回带有特殊标记的序列对输入 ID 列表
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的令牌列表中检索序列 ID。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个可选的序列对 ID 列表。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 返回单个序列的特殊标记掩码
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 返回序列对的特殊标记掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列对创建令牌类型 ID。这个方法用于创建用于区分不同序列的令牌类型 ID。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个可选的序列对 ID 列表。

        Returns:
            `List[int]`: 令牌类型 ID 列表。
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
        RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Define the separator token
        sep = [self.sep_token_id]
        # Define the classification token
        cls = [self.cls_token_id]

        # If token_ids_1 is not provided (single sequence case)
        if token_ids_1 is None:
            # Return a list of zeros of length equal to the sum of cls, token_ids_0, sep
            return len(cls + token_ids_0 + sep) * [0]
        
        # For sequence pairs case (token_ids_1 is provided)
        # Return a list of zeros of length equal to the sum of cls, token_ids_0, sep, sep, token_ids_1, sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```