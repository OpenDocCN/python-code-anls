# `.\models\fnet\tokenization_fnet.py`

```py
# coding=utf-8
# 上面的行声明了文件编码格式为 UTF-8，确保可以正确处理中文和其他特殊字符
# Copyright 2021 Google Research, Google AI, Google Brain and the HuggingFace Inc. team.
# 版权声明，指出了代码的版权归属及授权许可信息
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 授权许可，允许在符合许可的前提下使用本文件
# you may not use this file except in compliance with the License.
# 除非符合许可，否则禁止使用此文件
# You may obtain a copy of the License at
# 获取许可协议的副本，详见以下网址
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则依据 "AS IS" 原则发布软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的保证或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 详细内容请参阅许可协议，包括授权的特定语言和限制
""" Tokenization classes for FNet model."""
# 此行开始了对 FNet 模型的 tokenization 类的定义，是本文件的主题注释

import os
# 导入操作系统相关的模块
import unicodedata
# 导入处理 Unicode 数据的模块
from shutil import copyfile
# 导入复制文件的函数 copyfile
from typing import Any, Dict, List, Optional, Tuple
# 导入类型提示相关的功能，包括 Any, Dict, List, Optional, Tuple

import sentencepiece as spm
# 导入 SentencePiece 库，用于分词

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从 tokenization_utils 模块中导入 AddedToken 和 PreTrainedTokenizer 类
from ...utils import logging
# 从 utils 模块中导入 logging 模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}
# 定义词汇文件名字典，包含一个键值对，指定了词汇文件的名称为 "spiece.model"

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
}
# 预训练词汇文件映射字典，指定了不同模型与其对应的预训练词汇文件下载链接

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}
# 预训练位置嵌入大小字典，指定了不同模型的预训练位置嵌入大小为 512

SPIECE_UNDERLINE = "▁"
# 定义了 SentencePiece 使用的起始符号，这里是下划线 "▁"

class FNetTokenizer(PreTrainedTokenizer):
    """
    Construct an FNet tokenizer. Adapted from [`AlbertTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece). This tokenizer inherits from [`PreTrainedTokenizer`]
    which contains most of the main methods. Users should refer to this superclass for more information regarding those
    methods.
    """
    # FNetTokenizer 类的定义，继承自 PreTrainedTokenizer 类，实现 FNet 模型的分词器功能
    # 从 AlbertTokenizer 适配而来，基于 SentencePiece 实现
    # 初始化一个SentencePieceProcessor对象，用于加载和处理SentencePiece模型
    sp_model = SentencePieceProcessor()
    # 加载指定的SentencePiece模型文件，初始化tokenizer
    sp_model.Load(vocab_file)
    # 是否在tokenize时将输入文本转换为小写，默认为False
    self.do_lower_case = do_lower_case
    # 是否在tokenize时移除文本中的空格，默认为True
    self.remove_space = remove_space
    # 是否在tokenize时保留文本中的重音符号，默认为True
    self.keep_accents = keep_accents
    # 未知token，当输入的token不在词汇表中时使用，默认为"<unk>"
    self.unk_token = unk_token
    # 分隔token，用于多个序列合并时分隔不同的序列，默认为"[SEP]"
    self.sep_token = sep_token
    # 填充token，用于填充不同长度的序列，默认为"<pad>"
    self.pad_token = pad_token
    # 分类器token，用于序列分类时的特殊token，默认为"[CLS]"
    self.cls_token = cls_token
    # 掩码token，用于掩码语言模型训练时的特殊token，默认为"[MASK]"
    self.mask_token = mask_token
    # SentencePiece模型的额外参数，将会传递给SentencePieceProcessor.__init__()方法
    self.sp_model_kwargs = sp_model_kwargs if sp_model_kwargs is not None else {}
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """
    # 定义类变量，包含模型需要的文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果 mask_token 是字符串，则创建一个特殊的 AddedToken 对象
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 如果 cls_token 是字符串，则创建一个特殊的 AddedToken 对象
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串，则创建一个特殊的 AddedToken 对象
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 mask_token 是字符串，则创建一个特殊的 AddedToken 对象
        mask_token = AddedToken(mask_token, special=True) if isinstance(mask_token, str) else mask_token
        # 如果未提供 sp_model_kwargs，则设为默认空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 初始化参数赋值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 SentencePieceProcessor 初始化 sp_model，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回当前 sp_model 的词汇大小
        return len(self.sp_model)

    def get_vocab(self):
        # 生成词汇表，将 ID 映射到对应的词汇符号
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将已添加的特殊符号编码器合并到词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 获取对象状态的副本，去除 sp_model 属性以便序列化
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象状态，包括 sp_model 属性的重新初始化
        self.__dict__ = d

        # 兼容旧版本的处理
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 SentencePieceProcessor 重新初始化 sp_model 并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
    # 预处理文本，根据初始化时的设置进行处理
    def preprocess_text(self, inputs):
        if self.remove_space:
            # 如果需要移除空格，则去除首尾空格并用单个空格重新连接单词
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        # 替换文本中的特殊引号格式为标准双引号
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            # 如果不保留重音符号，则使用Unicode标准化处理文本
            outputs = unicodedata.normalize("NFKD", outputs)
            # 过滤掉所有组合字符，保留文本中的基本字符
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            # 如果需要转换为小写，则将文本全部转换为小写
            outputs = outputs.lower()

        return outputs

    # 使用SentencePiece模型进行分词
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 预处理文本
        text = self.preprocess_text(text)
        # 使用SentencePiece模型对文本进行编码，返回编码后的片段列表
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 处理以数字结尾且倒数第二个字符为逗号的片段
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    # 将Token转换为对应的ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    # 将ID转换为对应的Token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    # 从tokens序列中恢复成单个字符串
    # 参考自transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊token不会被SentencePiece模型解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    # 解码token_ids列表
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # 调用父类的 _decode 方法，解码 token_ids 为文本
        text = super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )
        # 模仿 Rust 分词器的行为：
        # 在 <unk> 后面不加空格
        if not spaces_between_special_tokens:
            text = text.replace("<unk> ", "<unk>")
        # 返回处理后的文本
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，构建用于序列分类任务的模型输入。一个 FNet 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        sep = [self.sep_token_id]  # 获取 SEP token 的 ID
        cls = [self.cls_token_id]  # 获取 CLS token 的 ID
        if token_ids_1 is None:
            return cls + token_ids_0 + sep  # 返回单个序列的特殊标记输入
        return cls + token_ids_0 + sep + token_ids_1 + sep  # 返回序列对的特殊标记输入

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从未添加特殊标记的标记列表中检索序列 ID。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1] 内：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        根据输入的序列列表创建 token_type_ids。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。

        Returns:
            None
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An FNet sequence
        pair mask has the following format: :

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence |
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
        # Define the separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a mask with all zeros for the first sequence part
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Otherwise, concatenate the lengths of cls, token_ids_0, sep with all zeros,
        # and concatenate the length of token_ids_1 and sep with all ones for the second sequence part
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if the save_directory exists; if not, log an error and return None
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Construct the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocab_file path is different from the output path and is a file,
        # copy the current vocab_file to the output vocab_file path
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If the current vocab_file does not exist, write the serialized sp_model proto to the output vocab_file
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # Return the path of the saved vocabulary file
        return (out_vocab_file,)
```