# `.\models\gemma\tokenization_gemma.py`

```
# coding=utf-8
# 定义编码格式为 UTF-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# 版权声明：2024 年 HuggingFace 公司团队。保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权使用此文件。

# you may not use this file except in compliance with the License.
# 您除非遵循许可证，否则不得使用此文件。

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本：

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，本软件按“原样”分发，不附带任何明示或暗示的保证或条件。

# See the License for the specific language governing permissions and
# limitations under the License.
# 详细了解许可证以了解权限和限制。

"""Tokenization classes for Gemma."""
# 导入 Gemma 的 Tokenization 类

import os
# 导入操作系统相关的模块
from shutil import copyfile
# 从 shutil 模块导入 copyfile 函数
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
# 导入类型检查相关的模块，以及一些数据结构

import sentencepiece as spm
# 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 tokenization_utils 模块中导入 AddedToken 和 PreTrainedTokenizer
from ...utils import logging
# 从 utils 模块导入 logging

if TYPE_CHECKING:
    pass
# 如果是类型检查阶段，则不执行任何操作

logger = logging.get_logger(__name__)
# 获取当前模块的 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}
# 定义词汇文件名的映射，vocab_file 对应的文件名是 tokenizer.model

SPIECE_UNDERLINE = "▁"
# 定义特殊字符 SPIECE_UNDERLINE 为 "▁"，用于表示词汇中的连接符

class GemmaTokenizer(PreTrainedTokenizer):
    """
    Construct a Gemma tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.
    """
    # GemmaTokenizer 类，继承自 PreTrainedTokenizer 类

    def __init__(
        self,
        # 初始化方法，接受以下参数：

    ```
    # 定义函数的参数和默认值
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记将被设置为此标记。
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<bos>"`):
            序列的起始标记，用于预训练过程中。也可以用作序列分类器的标记。
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<eos>"`):
            序列的结束标记。
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<pad>"`):
            特殊标记，用于使标记数组在批处理时具有相同的大小。在注意力机制或损失计算中将被忽略。
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            将传递给 `SentencePieceProcessor.__init__()` 方法的参数字典。
            可用于设置 SentencePiece 的参数，如启用子词正则化和采样参数等。
        add_bos_token (`bool`, *optional*, defaults to `True`):
            是否在序列的开头添加 `bos_token`。
        add_eos_token (`bool`, *optional*, defaults to `False`):
            是否在序列的末尾添加 `eos_token`。
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            是否在解码后清理空格，清理包括移除额外的空格等。
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            是否使用默认的系统提示。
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            是否在特殊标记之间添加空格。
    """

    # 定义词汇表文件名和模型输入名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，用于创建一个新的 LlamaTokenizer 对象
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        **kwargs,
    ):
        # 如果 sp_model_kwargs 为 None，则设为空字典，否则使用传入的参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 如果 bos_token 是字符串，则将其封装为一个 AddedToken 对象
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其封装为一个 AddedToken 对象
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则将其封装为一个 AddedToken 对象
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其封装为一个 AddedToken 对象
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        # 将传入的参数赋值给对象的属性
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt

        # 使用 SentencePieceProcessor 初始化 sp_model 对象，并加载给定的词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化函数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    # 复制自 transformers.models.llama.tokenization_llama.LlamaTokenizer.__getstate__
    def __getstate__(self):
        # 复制对象的当前状态
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，避免序列化时包含模型本身
        state["sp_model"] = None
        # 获取 sp_model 的序列化模型，并保存到状态中
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 复制自 transformers.models.llama.tokenization_llama.LlamaTokenizer.__setstate__
    def __setstate__(self, d):
        # 恢复对象的状态
        self.__dict__ = d
        # 使用 sp_model_kwargs 初始化 sp_model 对象，并从序列化的 proto 中加载模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    # 复制自 transformers.models.llama.tokenization_llama.LlamaTokenizer.vocab_size
    def vocab_size(self):
        """Returns vocab size"""
        # 返回词汇表的大小，即 sp_model 中词汇的数量
        return self.sp_model.get_piece_size()

    # 复制自 transformers.models.llama.tokenization_llama.LlamaTokenizer.get_vocab
    def get_vocab(self):
        """Returns vocab as a dict"""
        # 创建一个词汇表字典，将词汇索引映射为词汇
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将已添加的特殊 token 编码器合并到词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab
    # 返回经过 Gemma 分词器处理后的文本字符串，不添加前导空格
    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    # 从词汇表中将 token（字符串）转换为对应的 id，方法来自于 llama.tokenization_llama.LlamaTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    # 从词汇表中将 id（整数）转换为对应的 token（字符串），方法来自于 llama.tokenization_llama.LlamaTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        token = self.sp_model.IdToPiece(index)
        return token

    # 将 token_ids（整数列表）解码为字符串，可以选择跳过特殊 token 和在特殊 token 之间添加空格
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        sub_texts = []
        current_sub_text = []
        for ids in token_ids:
            if skip_special_tokens and ids in self.all_special_ids:
                continue
            if ids in self._added_tokens_decoder:
                if current_sub_text:
                    sub_texts.append(self.sp_model.decode(current_sub_text))
                sub_texts.append(self._added_tokens_decoder[ids].content)
                current_sub_text = []
            else:
                current_sub_text.append(ids)
        if current_sub_text:
            sub_texts.append(self.sp_model.decode(current_sub_text))

        if spaces_between_special_tokens:
            sub_texts = " ".join(sub_texts)
        else:
            sub_texts = "".join(sub_texts)

        return sub_texts

    # 将 token（字符串）序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in self._added_tokens_encoder:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # 保存词汇表的方法，来自 llama.tokenization_llama.LlamaTokenizer.save_vocabulary
    # 从给定的 `token_ids_0` 和 `token_ids_1` 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要，添加开头的 BOS (Beginning of Sentence) 标记
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        # 如果需要，添加结尾的 EOS (End of Sentence) 标记
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 构建包含特殊标记的输出序列
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果提供了 `token_ids_1`，再次构建包含特殊标记的输出序列
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # 从给定的 `token_ids_0` 和 `token_ids_1` 判断特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    # 从没有添加特殊标记的标记列表中获取序列ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
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
        # 如果已经包含特殊标记，则调用父类方法获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 根据是否添加起始（bos）和结束（eos）标记，初始化起始和结束标记ID列表
        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        # 如果没有第二个序列token_ids_1，则返回仅包含第一个序列的特殊标记掩码
        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        
        # 否则，返回包含两个序列的特殊标记掩码
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    # 从 `transformers.models.llama.tokenization_llama.LlamaTokenizer.create_token_type_ids_from_sequences` 复制过来的方法
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # 根据是否添加起始（bos）和结束（eos）标记，初始化起始和结束标记ID列表
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 初始化输出为全0的列表，长度为起始 + 第一个序列 + 结束的总长度
        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        # 如果有第二个序列token_ids_1，则设置第二个序列部分的token type ID为1
        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
```