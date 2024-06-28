# `.\models\reformer\tokenization_reformer.py`

```py
# coding=utf-8
# 声明脚本编码格式为 UTF-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# 版权声明，指出代码版权归属和授权信息
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License 2.0 授权许可使用本代码
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在上述链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不附带任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查阅许可证以了解具体的语言许可和限制
""" Tokenization class for model Reformer."""
# 用于 Reformer 模型的分词类

import os
# 导入操作系统相关功能模块
from shutil import copyfile
# 导入复制文件功能模块
from typing import Any, Dict, List, Optional, Tuple
# 导入类型提示模块

import sentencepiece as spm
# 导入 sentencepiece 库

from ...tokenization_utils import PreTrainedTokenizer
# 从 tokenization_utils 中导入 PreTrainedTokenizer 类
from ...utils import logging
# 从 utils 中导入 logging 模块


logger = logging.get_logger(__name__)
# 获取当前模块的 logger

SPIECE_UNDERLINE = "▁"
# 定义 SPIECE_UNDERLINE 符号为 "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}
# 定义词汇文件名映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/reformer-crime-and-punishment": (
            "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/spiece.model"
        )
    }
}
# 预训练词汇文件映射字典，包含模型名称和对应的词汇文件下载链接

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/reformer-crime-and-punishment": 524288,
}
# 预训练位置嵌入尺寸字典，包含模型名称和对应的位置嵌入大小


class ReformerTokenizer(PreTrainedTokenizer):
    """
    Construct a Reformer tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece) .

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # Reformer 分词器类，基于 SentencePiece，继承自 PreTrainedTokenizer
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        additional_special_tokens (`List[str]`, *optional*, defaults to `[]`):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    """

    # 定义类的常量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=[],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 初始化函数，设置实例变量
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        # 创建 SentencePieceProcessor 实例，并加载指定的词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法，传递特殊的 token 和 sp_model_kwargs
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回当前 SentencePieceProcessor 实例的词汇大小
        return self.sp_model.get_piece_size()
    def get_vocab(self) -> Dict[str, int]:
        # 创建一个词汇表字典，将词汇映射为其对应的 ID
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将额外添加的特殊token编码器内容合并到词汇表字典中
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 复制对象的当前状态
        state = self.__dict__.copy()
        # 置空sp_model字段，以准备进行对象的序列化
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象的状态
        self.__dict__ = d

        # 为了向后兼容性
        # 如果对象没有sp_model_kwargs属性，则创建一个空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用sp_model_kwargs参数重新初始化sp_model对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载vocab_file指定的词汇模型文件到sp_model对象中
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """接受一个字符串作为输入，并返回一个由单词/子词组成的列表（tokens）"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """将一个token（字符串）转换为其对应的ID，使用词汇表"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """将一个索引（整数）转换为其对应的token（字符串），使用词汇表"""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """将一系列token（字符串）转换为单个字符串"""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊token不使用sentencepiece模型进行解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不同，并且当前词汇表文件存在，则进行复制
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将sp_model的序列化模型内容写入目标路径文件中
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```