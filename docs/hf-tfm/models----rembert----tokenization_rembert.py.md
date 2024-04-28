# `.\transformers\models\rembert\tokenization_rembert.py`

```
# coding=utf-8 
## 在文件开始的地方，指定文件的编码格式为UTF-8，以防止中文或特殊字符出现乱码。
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
## 版权声明，版权归HuggingFace团队所有。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.                                                                                                                                                                                              
## 代码采用Apache许可2.0开源许可，可以根据许可证进行使用。
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing pere
## 可以在适用法律下或通过书面形式同意的情况下使用本软件。本软件按“原样”提供，没有提供明示或暗示的担保或条件。
## 详细阐述了如何配置该许可证
"""Tokenization classes for RemBERT."""   
## Tokenization类别用于RemBERT

import os
from shutil import copyfile
from typing import List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)
## 引入相关模块和库

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model"}
## 定义文件名与词汇表文件名的对应关系

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/sentencepiece.model",
    },
}
## 定义预训练好的字典文件的下载地址

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/rembert": 256,
}
## 字典映射表的键，指定了语言模型的名称和相应的位置嵌入尺寸大小。

class RemBertTokenizer(PreTrainedTokenizer):
## 创建了一个RemBertTokenizer类，继承自PreTrainedTokenizer类的所有方法和属性。

    """
    Construct a RemBERT tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
## 类的描述信息，告诉用户此Tokenizer类构建了一个RemBERT（RemBertForQuestionAnswering）的tokenizer。
## 该Tokenizer类继承自PreTrainedTokenizer类，PreTrainedTokenizer是一个存在于transformers库中的类，它包含了大多数主要方法。用户可以参考这个超类来获取这些方法的更多信息。
    # 定义一个函数，接受多个参数
        Args:
            vocab_file (`str`):
                [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
                contains the vocabulary necessary to instantiate a tokenizer. 用于实例化分词器的词汇表文件名
            bos_token (`str`, *optional*, defaults to `"[CLS]"`):
                The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
                <Tip>
    
                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.
    
                </Tip>
                开始序列标记，用于预训练期间。可以用作序列分类器标记。
    
            eos_token (`str`, *optional*, defaults to `"[SEP]"`):
                The end of sequence token.
    
                <Tip>
    
                When building a sequence using special tokens, this is not the token that is used for the end of sequence.
                The token used is the `sep_token`.
    
                </Tip>
                结束序列标记。
    
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
                未知标记。词汇表中不存在的标记无法转换为 ID，而会被设置为此标记。
    
            sep_token (`str`, *optional*, defaults to `"[SEP]"`):
                The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
                sequence classification or for a text and a question for question answering. It is also used as the last
                token of a sequence built with special tokens.
                分隔符标记，用于从多个序列构建序列，例如，用于序列分类的两个序列，或者用于问题回答的文本和问题。还用作使用特殊标记构建的序列的最后一个标记。
    
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
                用于填充的标记，例如，在批处理不同长度的序列时。
    
            cls_token (`str`, *optional*, defaults to `"[CLS]"`):
                The classifier token which is used when doing sequence classification (classification of the whole sequence
                instead of per-token classification). It is the first token of the sequence when built with special tokens.
                在进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）时使用的分类器标记。在使用特殊标记构建序列时，它是序���的第一个标记。
    
            mask_token (`str`, *optional*, defaults to `"[MASK]"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling. This is the token which the model will try to predict.
                用于屏蔽值的标记。在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
    
        Attributes:
            sp_model (`SentencePieceProcessor`):
                The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
                用于每个转换的 *SentencePiece* 处理器（字符串、标记和 ID）。
        """
    
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
        def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=True,
            keep_accents=True,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs,
        # 如果 mask_token 是字符串类型，则将其转换为 AddedToken 对象，使其在处理时包含前导空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 设置是否将文本转换为小写
        self.do_lower_case = do_lower_case
        # 设置是否移除空格
        self.remove_space = remove_space
        # 设置是否保留重音符号
        self.keep_accents = keep_accents
        # 设置词汇表文件路径
        self.vocab_file = vocab_file

        # 初始化 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor()
        # 从词汇表文件加载 SentencePiece 模型
        self.sp_model.Load(vocab_file)
        # 调用父类的构造函数初始化
        super().__init__(
            do_lower_case=do_lower_case,  # 是否转换为小写
            remove_space=remove_space,  # 是否移除空格
            keep_accents=keep_accents,  # 是否保留重音符号
            bos_token=bos_token,  # 开始标记
            eos_token=eos_token,  # 结束标记
            unk_token=unk_token,  # 未知标记
            sep_token=sep_token,  # 分隔标记
            pad_token=pad_token,  # 填充标记
            cls_token=cls_token,  # 类别标记
            mask_token=mask_token,  # 掩码标记
            **kwargs,  # 其他参数
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.sp_model)

    def get_vocab(self):
        # 获取词汇表，包括用户添加的标记
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 返回对象的状态（排除 sp_model）
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 设置对象的状态并重新加载 sp_model
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, sample=False):
        """对字符串进行分词处理。"""
        # 使用 SentencePiece 模型对文本进行分词
        pieces = self.sp_model.EncodeAsPieces(text)
        return pieces

    def _convert_token_to_id(self, token):
        """将标记（字符串）转换为 ID。"""
        # 使用词汇表将标记转换为 ID
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """将索引（整数）转换为标记（字符串）。"""
        # 使用词汇表将 ID 转换为标记
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        # 将标记序列转换为字符串
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 构建用于序列分类任务的模型输入
    # 通过拼接并添加特殊标记来构建输入序列
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # 定义分隔符和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果只有一个序列
        if token_ids_1 is None:
            # 返回 [CLS] + 序列 + [SEP]
            return cls + token_ids_0 + sep
        # 如果有两个序列
        # 返回 [CLS] + 序列A + [SEP] + 序列B + [SEP]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    # 获取特殊标记的掩码
    # 从没有添加特殊标记的标记列表中检索序列 ID
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        # 如果已经包含特殊标记
        if already_has_special_tokens:
            # 如果提供了第二个序列，则抛出错误
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of ids is already formatted with special tokens for the model.")
            # 返回 1 表示特殊标记，0 表示序列标记
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]
        
        # 如果有两个序列
        if token_ids_1 is not None:
            # 返回 [1, 0, 0, ..., 0, 1, 0, 0, ..., 0, 1]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 如果只有一个序列
        # 返回 [1, 0, 0, ..., 0, 1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    # 从序列中创建令牌类型 ID
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    def create_sequence_pair_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RemBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 分隔符
        sep = [self.sep_token_id]
        # 类别标识符
        cls = [self.cls_token_id]

        # 如果 token_ids_1 为 None，则只返回 mask 的第一部分（全为 0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回两部分组成的 mask，第一部分为 0，第二部分为 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果输入词汇表文件路径与输出词汇表文件路径不同且输入文件存在，则复制输入文件到输出文件路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果输入文件不存在，则将序列化后的 sp_model_proto 写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```