# `.\models\big_bird\tokenization_big_bird.py`

```
# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for BigBird."""


import os
import re
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入 logging 模块


logger = logging.get_logger(__name__)  # 获取当前模块的 logger

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}  # 定义词汇文件的名称映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": (
            "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model"
        ),
        "google/bigbird-base-trivia-itc": (
            "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model"
        ),
    }
}  # 预训练词汇文件的映射，包含模型名称及其对应的远程路径

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}  # 预训练位置嵌入的尺寸映射，包含模型名称及其对应的位置嵌入大小


class BigBirdTokenizer(PreTrainedTokenizer):
    """
    Construct a BigBird tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # BigBirdTokenizer 类，继承自 PreTrainedTokenizer，用于构建 BigBird 分词器，基于 SentencePiece
    pass  # 占位符，暂未实现额外的方法或属性，仅作为类声明的结尾
    # vocab_file 参数：指定 SentencePiece 文件的路径，该文件包含用于实例化分词器的词汇表
    # unk_token 参数（可选，默认为 "<unk>"）：未知标记，表示词汇表中不存在的词汇将被设置为此标记
    # bos_token 参数（可选，默认为 "<s>"）：序列开始标记
    # eos_token 参数（可选，默认为 "</s>"）：序列结束标记
    # pad_token 参数（可选，默认为 "<pad>"）：用于填充的标记，在处理不同长度的序列时使用
    # sep_token 参数（可选，默认为 "[SEP]"）：分隔符标记，用于构建多个序列的时候使用
    # mask_token 参数（可选，默认为 "[MASK]"）：掩码标记，在掩码语言建模（Masked Language Modeling）中使用，模型会尝试预测这些标记
    # cls_token 参数（可选，默认为 "[CLS]"）：分类器标记，用于序列分类任务中，表示序列的开始
    # sp_model_kwargs 参数（可选）：将传递给 SentencePieceProcessor.__init__() 方法的参数字典，
    # 可以用于配置 SentencePiece 的各种参数，例如启用子词正则化、设置采样参数等

    vocab_files_names = VOCAB_FILES_NAMES
    # vocab_files_names 变量：包含了预训练模型所需的词汇文件名的列表

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # pretrained_vocab_files_map 变量：包含了预训练模型对应的词汇文件路径的映射字典

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # max_model_input_sizes 变量：包含了预训练位置嵌入的最大输入尺寸的字典

    model_input_names = ["input_ids", "attention_mask"]
    # model_input_names 变量：包含了模型输入的名称列表，用于对应模型的输入要求

    prefix_tokens: List[int] = []
    # prefix_tokens 变量：用于存储特殊前缀标记的列表，初始化为空列表
    # 初始化函数，接受多个参数和关键字参数来配置词汇表和特殊标记
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果特殊标记是字符串，则将其转换为 AddedToken 对象，保留其空白字符处理设置
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        # Mask token 被视为普通单词，即包括其前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 如果未提供 sp_model_kwargs，则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 保存词汇表文件路径
        self.vocab_file = vocab_file

        # 创建 SentencePieceProcessor 对象，并加载词汇表文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法，传递特殊标记和其它关键字参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    # 返回词汇表大小，由 SentencePieceProcessor 对象提供
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    # 返回包含所有词汇及其对应 id 的字典，包括添加的特殊标记
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 返回对象的状态，用于序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None  # 移除 sp_model 对象，以免被序列化
        return state

    # 设置对象的状态，用于反序列化
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容性，如果缺少 sp_model_kwargs 属性，则设为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建 SentencePieceProcessor 对象并加载词汇表文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 对文本进行分词处理，返回由字符串组成的列表（标记）
    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        return self.sp_model.encode(text, out_type=str)

    # 将标记（字符串）转换为其对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)
    # 使用给定的索引在词汇表中将索引转换为对应的标记字符串
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 sentencepiece 模型将索引转换为对应的标记字符串
        token = self.sp_model.IdToPiece(index)
        return token

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string 复制而来
    # 将一系列标记字符串转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []  # 当前正在处理的子标记列表
        out_string = ""  # 输出的合并后的字符串
        prev_is_special = False  # 上一个标记是否为特殊标记
        for token in tokens:
            # 确保特殊标记不会使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "  # 添加空格来分隔特殊标记
                # 使用 sentencepiece 模型解码当前子标记列表，并加上当前特殊标记
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []  # 重置当前子标记列表
            else:
                current_sub_tokens.append(token)  # 将当前标记添加到当前子标记列表中
                prev_is_special = False
        # 将剩余的子标记列表使用 sentencepiece 模型解码，并添加到输出字符串中
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()  # 返回去掉两侧空格的输出字符串

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ):
        ) -> str:
        # 从 kwargs 中获取 use_source_tokenizer 参数，并设置到实例变量中
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 转换 token_ids 到 tokens 列表，跳过特殊标记（如果需要）
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 为了避免混合字节级别和 unicode 字符（例如字节级别 BPT），需要分别构建字符串以处理添加的标记和字节级别的 tokens
        # 参考：https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 如果 token 是添加的特殊标记
            if token in self.added_tokens_encoder:
                # 如果当前子文本不为空，则将其转换为字符串并添加到 sub_texts 中
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        # 将剩余的 current_sub_text 转换为字符串并添加到 sub_texts 中
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 模仿 Rust 分词器的行为：
        # 在 [MASK] 和 [SEP] 前不添加空格
        if spaces_between_special_tokens:
            # 使用正则表达式去除特殊标记前的空格
            text = re.sub(r" (\[(MASK|SEP)\])", r"\1", " ".join(sub_texts))
        else:
            text = "".join(sub_texts)

        # 根据 clean_up_tokenization_spaces 参数清理分词后的空格
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            # 清理分词后的空格
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径和目标文件路径不同且当前词汇表文件存在，则复制当前词汇表文件到目标文件路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将当前的词汇表内容写入目标文件路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的词汇表文件路径的元组
        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Big Bird sequence has the following format:

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
        # Check if only one sequence is provided
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # Define special tokens for the start and separation
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # Concatenate tokens for a pair of sequences
        return cls + token_ids_0 + sep + token_ids_1 + sep

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
        # If the tokens already have special tokens, delegate to the superclass
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate special tokens mask for a single sequence
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # Calculate special tokens mask for a pair of sequences
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from two sequences or a single sequence. Token type IDs are binary tensors where
        0 indicates the first sequence and 1 indicates the second sequence.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs.
        """
        # Initialize token type IDs for the first sequence
        token_type_ids = [0] * len(token_ids_0)
        # If token_ids_1 is provided, extend token type IDs to cover both sequences
        if token_ids_1 is not None:
            token_type_ids += [1] * len(token_ids_1)
        return token_type_ids
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second
        sequence | If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Separator token IDs for separating sequences
        sep = [self.sep_token_id]
        # Classification token ID indicating the start of a classification task
        cls = [self.cls_token_id]
        
        # If only one sequence (`token_ids_1` is `None`), return mask for the first sequence
        if token_ids_1 is None:
            # Return a list of zeros representing the mask for the first sequence
            return len(cls + token_ids_0 + sep) * [0]
        
        # If there are two sequences, return a combined mask for both sequences
        # Concatenate the length of cls + token_ids_0 + sep with zeros, then add the length of token_ids_1 + sep with ones
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
```