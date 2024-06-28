# `.\models\speecht5\tokenization_speecht5.py`

```
# coding=utf-8
# 上面是指定文件编码格式为 UTF-8

# 版权声明和许可证信息
# Copyright 2023 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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

"""Tokenization class for SpeechT5."""
# 上面是文件的简要描述和目的

# 引入必要的模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 引入 SentencePiece 库用于分词
import sentencepiece as spm

# 引入 tokenization_utils 模块中的 PreTrainedTokenizer 类
from ...tokenization_utils import PreTrainedTokenizer
# 引入 logging 模块中的日志记录器
from ...utils import logging
# 引入本地的 number_normalizer 模块中的 EnglishNumberNormalizer 类
from .number_normalizer import EnglishNumberNormalizer

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 定义词汇文件名常量
VOCAB_FILES_NAMES = {"vocab_file": "spm_char.model"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/spm_char.model",
        "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/spm_char.model",
        "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/spm_char.model",
    }
}

# 定义预训练模型的位置编码嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/speecht5_asr": 1024,
    "microsoft/speecht5_tts": 1024,
    "microsoft/speecht5_vc": 1024,
}


class SpeechT5Tokenizer(PreTrainedTokenizer):
    """
    Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 上面是 SpeechT5Tokenizer 类的描述和基本信息
    # 导入必要的库和模块
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) 文件的路径（通常具有 *.spm* 扩展名），
            包含实例化分词器所需的词汇表。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            序列的开始标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列的结束标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。当词汇表中没有某个词时，将该词转换为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，在将不同长度的序列进行批处理时使用。
        normalize (`bool`, *optional*, defaults to `False`):
            是否将文本中的数字量转换为其英文拼写的对应词。
        sp_model_kwargs (`dict`, *optional*):
            将传递给 `SentencePieceProcessor.__init__()` 方法的参数。可以用于设置 SentencePiece 的一些选项，
            如启用子词正则化 (`enable_sampling`)、nbest_size 参数等。

              - `enable_sampling`: 启用子词正则化。
              - `nbest_size`: 对于unigram的采样参数。对于BPE-Dropout无效。

                - `nbest_size = {0,1}`: 不执行采样。
                - `nbest_size > 1`: 从 nbest_size 个结果中进行采样。
                - `nbest_size < 0`: 假设 nbest_size 为无限大，使用前向过滤和后向采样算法从所有假设（lattice）中采样。

              - `alpha`: unigram 采样的平滑参数，以及 BPE-dropout 合并操作的 dropout 概率。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和ID）的 *SentencePiece* 处理器。
    """

    # 定义一些常量和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        normalize=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # 初始化函数，用于实例化一个新的 SentencePieceTokenizer 对象
    ) -> None:
        # 初始化函数，设置参数并加载 SentencePiece 模型
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs  # 如果未提供 sp_model_kwargs，则初始化为空字典
        self.vocab_file = vocab_file  # 设置词汇文件路径
        self.normalize = normalize  # 设置是否进行文本归一化
        self._normalizer = None  # 初始化归一化器为 None

        # 使用给定的参数初始化 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)  # 加载指定的词汇文件

        # 调用父类的初始化方法，传递相关参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            normalize=normalize,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 准备文本用于分词处理
        normalize = kwargs.pop("normalize", self.normalize)  # 获取归一化参数，如果未指定则使用实例变量的值
        if is_split_into_words:
            text = " " + text  # 如果文本已经分成单词，则在文本前加空格
        if normalize:
            text = self.normalizer(text)  # 如果需要归一化，则对文本进行归一化处理
        return (text, kwargs)  # 返回处理后的文本和剩余的 kwargs 参数

    @property
    def vocab_size(self):
        # 返回词汇表大小，即 SentencePiece 模型中的词汇数量
        return self.sp_model.get_piece_size()

    @property
    def normalizer(self):
        # 返回归一化器对象，如果未初始化则创建一个英文数字归一化器
        if self._normalizer is None:
            self._normalizer = EnglishNumberNormalizer()
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        # 设置归一化器对象
        self._normalizer = value

    def get_vocab(self):
        # 返回词汇表，将词汇 ID 映射为对应的词汇（字符串形式）
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)  # 将额外的特殊标记也加入词汇表
        return vocab

    def __getstate__(self):
        # 获取对象的状态，用于序列化
        state = self.__dict__.copy()
        state["sp_model"] = None  # 将 sp_model 设为 None，以免在序列化时保存 SentencePieceProcessor 对象
        return state

    def __setstate__(self, d):
        # 设置对象的状态，用于反序列化
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建 SentencePieceProcessor 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # 对文本进行分词处理，返回分词结果（字符串列表）
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将词汇（token）转换为对应的 ID
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 ID 转换为对应的词汇（token）
        token = self.sp_model.IdToPiece(index)
        return token

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) into a single string."""
        # 初始化空列表用于存储当前子 token 序列
        current_sub_tokens = []
        # 初始化输出字符串
        out_string = ""
        # 初始化标志来跟踪上一个 token 是否是特殊 token
        prev_is_special = False
        # 遍历 tokens 序列
        for token in tokens:
            # 检查当前 token 是否是特殊 token
            if token in self.all_special_tokens:
                # 如果当前 token 是特殊 token 并且上一个 token 不是特殊 token，则添加空格
                if not prev_is_special:
                    out_string += " "
                # 解码当前子 token 序列并添加当前 token 到输出字符串
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # 更新标志表明当前 token 是特殊 token
                prev_is_special = True
                # 重置当前子 token 序列
                current_sub_tokens = []
            else:
                # 将当前 token 添加到当前子 token 序列中
                current_sub_tokens.append(token)
                # 更新标志表明当前 token 不是特殊 token
                prev_is_special = False
        # 将剩余的子 token 序列解码并添加到输出字符串
        out_string += self.sp_model.decode(current_sub_tokens)
        # 返回去除首尾空格的输出字符串
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        # 如果只有一个输入序列，则在末尾添加 eos_token_id 并返回
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # 如果有两个输入序列，将它们连接并在末尾添加 eos_token_id 后返回
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        # 如果输入的 token_ids_0 已经包含特殊 token，直接调用父类的方法并返回结果
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 后缀添加 1 用于标识特殊 token
        suffix_ones = [1]
        # 如果只有一个输入序列，返回长度为 token_ids_0 的零列表加上后缀
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + suffix_ones
        # 如果有两个输入序列，返回长度为 token_ids_0 和 token_ids_1 的零列表加上后缀
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出词汇表文件路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将当前 sentencepiece 模型的序列化模型写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```