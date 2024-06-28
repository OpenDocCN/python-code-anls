# `.\models\speech_to_text\tokenization_speech_to_text.py`

```py
# coding=utf-8
# 设置脚本编码格式为 UTF-8

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归 HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 遵循 Apache License 2.0 版本，允许在特定条件下使用本代码

# you may not use this file except in compliance with the License.
# 除非符合许可证的条件，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在上述 License 链接获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证详细信息请访问 http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 在适用法律要求或书面同意的情况下，本软件按“原样”分发，不提供任何担保或条件

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证，了解适用的语言、权限和限制

"""Tokenization classes for Speech2Text."""
# 用于 Speech2Text 的分词类

import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece  # 导入 sentencepiece 库

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器基类
from ...utils import logging  # 导入日志模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

SPIECE_UNDERLINE = "▁"  # 定义表示词组起始的特殊符号

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "spm_file": "sentencepiece.bpe.model",  # sentencepiece 模型文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/s2t-small-librispeech-asr": (
            "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/vocab.json"
        ),
    },
    "spm_file": {
        "facebook/s2t-small-librispeech-asr": (
            "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/sentencepiece.bpe.model"
        )
    },
}

MAX_MODEL_INPUT_SIZES = {
    "facebook/s2t-small-librispeech-asr": 1024,  # 模型输入的最大长度
}

MUSTC_LANGS = ["pt", "fr", "ru", "nl", "ro", "it", "es", "de"]  # MUSTC 语言列表

LANGUAGES = {"mustc": MUSTC_LANGS}  # 支持的语言映射，例如 "mustc" 对应的语言列表为 MUSTC_LANGS


class Speech2TextTokenizer(PreTrainedTokenizer):
    """
    Construct an Speech2Text tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    """
    # 构造一个 Speech2Text 分词器，继承自 PreTrainedTokenizer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 调用父类的构造方法初始化分词器
    # 定义一个类，用于处理预训练模型的tokenizer，继承自`PreTrainedTokenizer`
    class PreTrainedTokenizer:
        
        # 定义类属性，指定用于加载词汇表文件的名称
        vocab_files_names = VOCAB_FILES_NAMES
        # 指定预训练模型的词汇表文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 指定模型的最大输入尺寸
        max_model_input_sizes = MAX_MODEL_INPUT_SIZES
        # 定义输入名称列表，这些名称将在模型输入时使用
        model_input_names = ["input_ids", "attention_mask"]
        
        # 初始化方法，设置tokenizer的各种参数
        def __init__(
            self,
            vocab_file,
            spm_file,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            do_upper_case=False,
            do_lower_case=False,
            tgt_lang=None,
            lang_codes=None,
            additional_special_tokens=None,
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
        ):
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 初始化参数 `sp_model_kwargs`，如果为 None 则设为空字典，否则使用传入的参数值

        self.do_upper_case = do_upper_case
        self.do_lower_case = do_lower_case
        # 设置是否进行大写和小写处理的标志

        self.encoder = load_json(vocab_file)
        # 加载并存储从 JSON 文件中读取的编码器字典

        self.decoder = {v: k for k, v in self.encoder.items()}
        # 创建解码器字典，反转编码器字典中的键值对

        self.spm_file = spm_file
        # 存储 SentencePiece 模型文件路径

        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)
        # 使用 SentencePiece 模型文件和参数初始化 sp_model 对象

        if lang_codes is not None:
            self.lang_codes = lang_codes
            # 存储语言代码

            self.langs = LANGUAGES[lang_codes]
            # 获取对应语言代码的语言列表

            self.lang_tokens = [f"<lang:{lang}>" for lang in self.langs]
            # 为每种语言生成特定格式的标记

            self.lang_code_to_id = {lang: self.sp_model.PieceToId(f"<lang:{lang}>") for lang in self.langs}
            # 创建语言到其对应 ID 的映射

            if additional_special_tokens is not None:
                additional_special_tokens = self.lang_tokens + additional_special_tokens
            else:
                additional_special_tokens = self.lang_tokens
            # 添加额外的特殊标记，包括语言标记

            self._tgt_lang = tgt_lang if tgt_lang is not None else self.langs[0]
            # 设置目标语言，默认为语言列表中的第一个

            self.set_tgt_lang_special_tokens(self._tgt_lang)
            # 设置目标语言的特殊标记
        else:
            self.lang_code_to_id = {}
            # 若未提供语言代码，则初始化为空字典

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            do_upper_case=do_upper_case,
            do_lower_case=do_lower_case,
            tgt_lang=tgt_lang,
            lang_codes=lang_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        # 调用父类的构造函数，传递参数以初始化基类

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
        # 返回编码器的词汇表大小

    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
        # 返回扩展后的完整词汇表，包括已添加的特殊标记

    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang
        # 返回当前目标语言代码

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang) -> None:
        self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(new_tgt_lang)
        # 设置新的目标语言，并更新特殊标记设置

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[eos, tgt_lang_code] and suffix=[eos]."""
        lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [lang_code_id]
        # 根据目标语言设置特殊标记，包括前缀部分的语言代码

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)
        # 使用 SentencePiece 模型对文本进行分词处理，返回分词后的字符串列表

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])
        # 将标记转换为对应的 ID，如果不在词汇表中则返回未知标记的 ID

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)
        # 将索引转换为对应的标记，使用解码器进行映射，未知索引返回未知标记
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 初始化当前子词列表和输出字符串
        current_sub_tokens = []
        out_string = ""
        # 遍历每个 token
        for token in tokens:
            # 检查当前 token 是否是特殊 token
            if token in self.all_special_tokens:
                # 解码当前子词列表成字符串，根据需求转换大小写，然后添加当前 token 到结果字符串中
                decoded = self.sp_model.decode(current_sub_tokens)
                out_string += (decoded.upper() if self.do_upper_case else decoded) + token + " "
                # 重置当前子词列表
                current_sub_tokens = []
            else:
                # 将当前 token 加入当前子词列表中
                current_sub_tokens.append(token)
        # 处理剩余的子词列表并添加到输出字符串中
        decoded = self.sp_model.decode(current_sub_tokens)
        out_string += decoded.upper() if self.do_upper_case else decoded
        # 返回处理后的输出字符串，去除末尾的空格
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        # 如果只有一个 token 序列，将其加上前缀 tokens 和结束 token 后返回
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # 对于处理序列对的情况，尽管本方法不期望处理，但保留对序列对的处理逻辑以保持 API 的一致性
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

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

        if already_has_special_tokens:
            # 如果输入已经包含特殊 tokens，则调用父类方法获取特殊 token 的掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 前缀 tokens 全部标记为 1
        prefix_ones = [1] * len(self.prefix_tokens)
        # 后缀 tokens 只有一个标记为 1
        suffix_ones = [1]
        if token_ids_1 is None:
            # 如果只有一个 token 序列，前缀为 1，其余为 0，后缀为 1
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # 对于序列对，前缀为 1，两个序列的内容为 0，后缀为 1
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def __getstate__(self) -> Dict:
        # 复制当前对象的状态字典
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，以便于对象的序列化
        state["sp_model"] = None
        # 返回修改后的状态字典
        return state
    # 定义对象的 __setstate__ 方法，用于从字典 d 恢复对象状态
    def __setstate__(self, d: Dict) -> None:
        # 将对象的 __dict__ 属性设置为字典 d，从而恢复对象的状态
        self.__dict__ = d

        # 为了向后兼容性而添加，如果对象没有属性 "sp_model_kwargs"，则设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用指定的 spm_file 和 sp_model_kwargs 加载 SentencePiece 模型
        self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)

    # 保存词汇表到指定目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 将 save_directory 转换为 Path 对象
        save_dir = Path(save_directory)
        # 断言 save_dir 是一个目录，否则抛出异常
        assert save_dir.is_dir(), f"{save_directory} should be a directory"

        # 构建词汇表文件和 SentencePiece 模型文件的保存路径
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        # 将编码器(encoder)保存为 JSON 文件到 vocab_save_path
        save_json(self.encoder, vocab_save_path)

        # 如果当前 spm_file 的绝对路径与 spm_save_path 不同，并且 spm_file 是一个文件，则复制 spm_file 到 spm_save_path
        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        # 如果 spm_file 不是文件，则将 sp_model 序列化后的模型内容写入到 spm_save_path
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的 vocab 和 spm 文件的路径组成的元组
        return (str(vocab_save_path), str(spm_save_path))
# 根据指定的参数加载 SentencePiece 模型处理器
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 使用给定的参数初始化 SentencePiece 模型处理器
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 加载指定路径下的 SentencePiece 模型
    spm.Load(str(path))
    # 返回加载后的 SentencePiece 模型处理器
    return spm


# 加载指定路径下的 JSON 文件并返回解析后的 Python 对象
def load_json(path: str) -> Union[Dict, List]:
    # 使用只读模式打开指定路径下的 JSON 文件
    with open(path, "r") as f:
        # 解析 JSON 文件内容为 Python 字典或列表
        return json.load(f)


# 将数据保存为 JSON 格式到指定路径下的文件
def save_json(data, path: str) -> None:
    # 以写入模式打开指定路径下的文件
    with open(path, "w") as f:
        # 将数据以带缩进的 JSON 格式写入文件
        json.dump(data, f, indent=2)
```