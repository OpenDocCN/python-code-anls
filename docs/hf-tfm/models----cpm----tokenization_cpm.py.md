# `.\models\cpm\tokenization_cpm.py`

```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import SPIECE_UNDERLINE, logging  # 导入特定工具和库


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}  # 定义词汇文件的名称字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    }
}


class CpmTokenizer(PreTrainedTokenizer):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练词汇文件映射

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(  # 调用父类的初始化方法
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.vocab_file = vocab_file  # 设置词汇文件路径
        self.do_lower_case = do_lower_case  # 是否将输入转换为小写
        self.remove_space = remove_space  # 是否移除空格
        self.keep_accents = keep_accents  # 是否保留重音符号

        if sp_model_kwargs is None:
            sp_model_kwargs = {}  # 如果未提供参数，默认为空字典
        self.sp_model_kwargs = sp_model_kwargs  # 设置 sentencepiece 模型的参数

    @property
    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.vocab_size
    def vocab_size(self):
        return len(self.sp_model)  # 返回 sentencepiece 模型的词汇量大小

    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.get_vocab
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # 创建并返回词汇表
        vocab.update(self.added_tokens_encoder)  # 将额外添加的特殊标记加入词汇表
        return vocab

    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()  # 复制当前对象的字典表示
        state["sp_model"] = None  # 将 sentencepiece 模型设为 None
        return state

    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d  # 恢复对象的状态字典

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}  # 处理旧版本兼容性

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)  # 初始化 sentencepiece 模型
        self.sp_model.Load(self.vocab_file)  # 加载指定的词汇文件
    # 从 XLNetTokenizer 类中复制的方法，用于预处理文本输入
    def preprocess_text(self, inputs):
        # 如果设置了 remove_space 标志，去除输入文本两端空格并用单个空格连接
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        # 替换特定的引号符号，将 `` 和 '' 替换为双引号 "
        outputs = outputs.replace("``", '"').replace("''", '"')

        # 如果不保留重音符号，进行 Unicode 标准化处理
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            # 过滤掉组合字符，保留单个字符
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        # 如果进行小写处理，将输出文本转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    # 从 XLNetTokenizer 类中复制的方法，用于将文本分词为子词列表
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 使用 preprocess_text 方法预处理文本
        text = self.preprocess_text(text)
        # 使用 sp_model 对象对文本进行编码，得到编码后的片段列表
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        # 遍历编码后的片段列表
        for piece in pieces:
            # 如果片段长度大于1且以逗号结尾且倒数第二个字符是数字，进行特殊处理
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 使用 sp_model.EncodeAsPieces 方法对片段进行进一步分解
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                # 如果原片段不以 SPIECE_UNDERLINE 开头且当前片段以此开头，调整处理
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                # 将处理后的片段加入到新片段列表中
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                # 否则直接将片段加入到新片段列表中
                new_pieces.append(piece)

        return new_pieces

    # 从 XLNetTokenizer 类中复制的方法，用于将 token 转换为其在词汇表中的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用 sp_model 对象的 PieceToId 方法将 token 转换为对应的 id
        return self.sp_model.PieceToId(token)

    # 从 XLNetTokenizer 类中复制的方法，用于将 id 转换为其在词汇表中的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 sp_model 对象的 IdToPiece 方法将 index 转换为对应的 token
        return self.sp_model.IdToPiece(index)

    # 从 XLNetTokenizer 类中复制的方法，用于将 token 列表转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 将 token 列表连接为一个字符串，并替换 SPIECE_UNDERLINE 为空格，去除首尾空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 从 XLNetTokenizer 类中复制的方法，用于构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        # 这个方法未完整给出，需要继续补充完整以符合原代码功能
    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate token type IDs from a sequence or a pair of sequences. XLNet uses token type IDs to distinguish
        between sequences in a pair.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs where each ID corresponds to a sequence or a pair of sequences.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
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
        sep = [self.sep_token_id]  # 分隔符列表，包含 XLNet 分隔符的 ID
        cls_segment_id = [2]  # 表示 XLNet 中的类别分割 ID

        if token_ids_1 is None:
            # 如果 token_ids_1 为 None，则返回仅包含第一个序列部分的 mask（全为 0）
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        # 否则，返回包含两个序列部分的 mask，第一个序列部分为 0，第二个序列部分为 1，最后是类别分割 ID
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    # Copied from transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            # 如果当前词汇表文件与目标路径不同并且当前词汇表文件存在，则复制当前词汇表文件到目标路径
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            # 否则，将当前词汇表文件的序列化模型写入目标路径
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)  # 调用父类的 _decode 方法获取文本
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")  # 处理文本中的特殊字符替换
        return text
```