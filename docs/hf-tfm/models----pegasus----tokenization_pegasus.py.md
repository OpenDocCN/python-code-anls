# `.\models\pegasus\tokenization_pegasus.py`

```py
# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
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

# 导入标准库 os
import os
# 从 shutil 库中导入 copyfile 函数
from shutil import copyfile
# 从 typing 库中导入类型提示 Any, Dict, List, Optional, Tuple
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，作为 spm 的别名
import sentencepiece as spm

# 从 tokenization_utils 中导入 AddedToken, PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 utils 中导入 logging 模块
from ...utils import logging

# 定义 SPIECE_UNDERLINE 常量
SPIECE_UNDERLINE = "▁"

# 定义 VOCAB_FILES_NAMES 字典常量
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义 PRETRAINED_VOCAB_FILES_MAP 字典常量
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"}
}

# 定义 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 字典常量
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}

# 获取 logger 对象
logger = logging.get_logger(__name__)


# TODO ArthurZ refactor this to only use the added_tokens_encoder
# 定义 PegasusTokenizer 类，继承自 PreTrainedTokenizer
class PegasusTokenizer(PreTrainedTokenizer):
    r"""
    Construct a PEGASUS tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    """

    # 设置类变量 vocab_files_names
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置类变量 pretrained_vocab_files_map
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设置类变量 max_model_input_sizes
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置类变量 model_input_names
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        offset=103,  # entries 2 - 104 are only used for pretraining
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(
            # 传递给父类的参数
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        # 设置实例变量 offset
        self.offset = offset  # entries 2 - 104 are only used for pretraining
        # 设置实例变量 sp_model_kwargs
        self.sp_model_kwargs = sp_model_kwargs if sp_model_kwargs is not None else {}
        # 加载 SentencePieceProcessor 对象到 self.sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到 self.vocab_file
        self.sp_model.Load(self.vocab_file)

    # 定义 vocab_size 属性方法，返回词汇表大小
    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + self.offset

    # 定义 get_vocab 方法，返回词汇表的字典
    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 定义 __getstate__ 方法，返回对象状态的字典表示，忽略 sp_model
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 定义 __setstate__ 方法，设置对象状态，重新加载 sp_model 和 vocab_file
    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        # 向后兼容性处理
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 加载 SentencePieceProcessor 到 self.sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇表文件到 self.vocab_file
        self.sp_model.Load(self.vocab_file)
    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # 使用 sentencepiece 模型对输入文本进行编码，返回字符串的列表（标记）
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        # 使用 sentencepiece 模型将 token（字符串）转换为对应的 id
        sp_id = self.sp_model.piece_to_id(token)
        return sp_id + self.offset

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocab."""
        # 如果 index 小于 offset，则直接使用 sentencepiece 模型将 index 转换为 token
        if index < self.offset:
            return self.sp_model.IdToPiece(index)
        # 否则，减去 offset 后再转换为 token
        token = self.sp_model.IdToPiece(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊的 token 不会被 sentencepiece 模型解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def num_special_tokens_to_add(self, pair=False):
        """Just EOS"""
        # 返回要添加的特殊 token 数量（这里只有 EOS）
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # 一次性创建所有特殊 token 的集合
        all_special_ids.remove(self.unk_token_id)  # <unk> 只有在某些情况下是特殊的

        # 创建一个 mask 列表，标记哪些 token 是特殊 token
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            # 如果已经有特殊 token，则直接调用 _special_token_mask 处理 token_ids_0
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            # 如果没有第二个 token 列表，对 token_ids_0 进行处理并添加一个额外的特殊 token
            return self._special_token_mask(token_ids_0) + [1]
        else:
            # 否则，合并 token_ids_0 和 token_ids_1 后处理，并添加一个额外的特殊 token
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
        and adding special tokens. A PEGASUS sequence has the following format, where `X` represents the sequence:

        - single sequence: `X </s>`
        - pair of sequences: `A B </s>` (not intended use)

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果只有一个序列，直接在末尾添加结束符号的特殊token
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        
        # 如果有两个序列，将两个序列连接起来，并在最后添加结束符号的特殊token
        # 尽管不推荐使用两个序列，但为了API的一致性保留了对两个序列的处理逻辑
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a directory. If the directory does not exist, an error is logged.

        Args:
            save_directory (str):
                Directory path where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved vocabulary file.

        Returns:
            `Tuple[str]`: Tuple containing the path to the saved vocabulary file.
        """
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 拼接输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 如果当前词汇表文件路径与目标文件路径不同且当前词汇表文件存在，则复制当前文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的模型写入目标路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        
        # 返回保存的词汇表文件路径
        return (out_vocab_file,)
```