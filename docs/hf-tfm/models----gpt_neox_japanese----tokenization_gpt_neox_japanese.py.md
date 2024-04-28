# `.\models\gpt_neox_japanese\tokenization_gpt_neox_japanese.py`

```
# coding=utf-8
# 设置脚本文件编码格式为UTF-8
# 版权声明
# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
# 版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可
# 只有在符合许可证的情况下才可以使用此文件
# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 如果没有按照适用法律规定或书面同意，则根据许可证分发的软件是根据“原样”分发的，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 有关特定语言的许可证来管理权限和限制条件
"""Tokenization classes for GPTNeoXJapanese."""
# 为GPTNeoXJapanese提供分词类
import collections
# 引入collections模块
import json
# 引入json模块
import os
# 引入os模块
import re
# 引入re模块
from typing import Optional, Tuple
# 从typing模块中引入Optional, Tuple类型

import numpy as np
# 引入numpy模块，命名为np

from ...tokenization_utils_fast import PreTrainedTokenizer
# 从tokenization_utils_fast模块中引入PreTrainedTokenizer
from ...utils import logging
# 从utils模块中引入logging

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}
# 定义VOCAB_FILES_NAMES为一个包含'vocab_file'和'emoji_file'的字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/vocab.txt",
    },
    "emoji_file": {
        "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/emoji.json",
    },
}
# 设置预训练时的词汇文件映射关系

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "abeja/gpt-neox-japanese-2.7b": 2048,
}
# 设置预训练时的位置嵌入尺寸

def load_vocab_and_emoji(vocab_file, emoji_file):
    """Loads a vocabulary file and emoji file into a dictionary."""
    # 加载词汇文件和表情符号文件到字典中
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())
    # 以utf-8编码以只读方式打开表情符号文件，将文件内容加载为json格式的数据

    vocab = collections.OrderedDict()
    # 创建有序字典
    raw_vocab = collections.OrderedDict()
    # 创建有序字典
    ids_to_tokens = collections.OrderedDict()
    # 创建有序字典
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()
    # 以utf-8编码以只读方式打开词汇文件，将行数据逐行读入token
    token = [[t.rstrip("\n")] if (t == "," or "," not in t) else t.rstrip("\n").split(",") for t in token]
    # 对读入的token进行处理
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx
    # 遍历处理后的token进行处理，创建字典

    return vocab, raw_vocab, ids_to_tokens, emoji
    # 返回词汇表、原始词汇表、标记到词汇的映射、表情符号

class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    # 通过PreTrainedTokenizer继承创建GPTNeoXJapaneseTokenizer类
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] and is based on Japanese special Sub-Word-Encoding that is
    used in this repository (https://github.com/tanreinama/Japanese-BPEEncoder_V2). Check the repository for details.
    Japanese has a relatively large vocabulary and there is no separation between words. Furthermore, the language is a
    combination of hiragana, katakana, and kanji, and variants such as "1" and "①" are often used. In order to cope
    with these, this tokenizer has the following features
    - Subword-by-subword segmentation, which is intermediate between byte strings and morphological analysis.
    # 这个分词器继承自[`PreTrainedTokenizer`]，并且基于日本特有的子词编码，该编码在此存储库中使用（https://github.com/tanreinama/Japanese-BPEEncoder_V2）。查看存储库以获取详细信息。
    # 日语词汇相对较多，且单词之间没有分隔。此外，语言是平假名、片假名和汉字的组合，还经常使用“1”和“①”等变体。为了应对这些情况，这个分词器具有以下特征
    # - 逐个子字的分割，介于字节串和形态分析之间。
    # BPEs 是为每个汉字、平假名和片假名字符创建的，不会跨字符类型，比如汉字+平假名或平假名+片假名。
    # 这是一个基于全字节编码的模型，不需要 <unk> 标记。
    # 与 UTF 编码无关，如2字节和3字节字符。
    # 异形文字被转换为相同的 token_id。
    # 表情符号和表情符号被分组为12种特殊标签。
    
    Example:
    
    # 导入 GPTNeoXJapaneseTokenizer 类
    >>> from transformers import GPTNeoXJapaneseTokenizer
    
    # 使用预训练模型 'abeja/gpt-neox-japanese-2.7b' 初始化 tokenizer 对象
    >>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    # 你可以确认 "慶応" 和 "慶應" 都被编码为 17749
    >>> tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"]
    [30014, 26883, 26638, 27228, 25, 26650, 31732, 31679, 27809, 26638, 17749, 31592, 17749, 31593, 321, 1281]
    
    # "慶応" 和 "慶應" 都被解码为 "慶応"
    >>> tokenizer.decode(tokenizer("吾輩は猫である🐯。実は慶応(慶應)大学出身")["input_ids"])
    '吾輩は猫である🐯。実は慶応(慶応)大学出身'
    
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        emoji_file (`str`):
            表情符号文件的路径。
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            未知 token。词汇表中没有的 token 无法转换为 ID，会被设置为这个 token。
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            用于填充的 token。
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            序列开始的 token。
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            序列结束的 token。
        do_clean_text (`bool`, *optional*, defaults to `False`):
            是否对文本进行清理，包括 URL、EMAIL、TEL、日文日期和日文价格。
    
    """
    
    # 定义一些类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file,
        emoji_file,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        do_clean_text=False,
        **kwargs,
    ):
        # 如果词汇文件不存在，引发 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 如果表情文件不存在，引发 ValueError 异常
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find a emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 初始化参数
        self.do_clean_text = do_clean_text
        # 加载词汇表和表情信息
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        # 创建 SubWordJapaneseTokenizer 对象
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )
        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_clean_text=do_clean_text,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.raw_vocab)

    def get_vocab(self):
        # 返回词汇表以及添加的标记编码的字典
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # 使用子词级别的分词器对文本进行分词
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引转换为对应的 token
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列 token 转换为单个字符串
        out_string = "".join(tokens).strip()
        return out_string

    @property
    def default_chat_template(self):
        """
        A simple chat template that just adds BOS/EOS tokens around messages while discarding role information.
        """
        # 返回默认的聊天模板，并发出警告
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return (
            "{% for message in messages %}"
            "{{ bos_token + eos_token + message.content + eos_token }}"
            "{% endfor %}"
            "{% if add_generation_prompt %} {{ bos_token + eos_token }} {% endif %}"
        )
    # 保存词汇表和表情符号到指定目录，返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 判断保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
            # 构建表情符号文件路径
            emoji_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["emoji_file"]
            )
        else:
            # 构建词汇表文件路径
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            # 构建表情符号文件路径
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        # 打开词汇表文件，写入词汇表内容
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的索引和词汇
            for token_index, token in self.ids_to_tokens.items():
                # 检查索引是否连续
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇
                writer.write(",".join(token) + "\n")
                index += 1
        # 打开表情符号文件，写入表情符号内容
        with open(emoji_file, "w", encoding="utf-8") as writer:
            # 将表情符号内容写入文件
            json.dump(self.emoji, writer)
        # 返回保存的词汇表文件路径和表情符号文件路径
        return vocab_file, emoji_file
class SubWordJapaneseTokenizer(object):
    """
    https://github.com/tanreinama/Japanese-BPEEncoder_V2 This tokenizer class is under MIT Lisence according to the
    original repository.

    MIT License

    Copyright (c) 2020 tanreinama

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, vocab, ids_to_tokens, emoji):
        self.vocab = vocab  # same as swe
        self.ids_to_tokens = ids_to_tokens  # same as bpe
        self.emoji = emoji
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])
        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}")
        self.content_repatter4 = re.compile(
            r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter5 = re.compile(
            r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|\u32ff)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
        )
        self.content_repatter6 = re.compile(
            r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*"
        )
        keisen = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
        blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})

    def __len__(self):
        return len(self.ids_to_tokens)
    # 清洗文本内容，替换特定模式的内容为指定标记
    def clean_text(self, content):
        # 使用正则表达式1替换内容中的URL为"<URL>"
        content = self.content_repatter1.sub("<URL>", content)
        # 使用正则表达式2替换内容中的EMAIL为"<EMAIL>"
        content = self.content_repatter2.sub("<EMAIL>", content)
        # 使用正则表达式3替换内容中的TEL为"<TEL>"
        content = self.content_repatter3.sub("<TEL>", content)
        # 使用正则表达式4替换内容中的DATE为"<DATE>"
        content = self.content_repatter4.sub("<DATE>", content)
        # 使用正则表达式5替换内容中的DATE为"<DATE>"
        content = self.content_repatter5.sub("<DATE>", content)
        # 使用正则表达式6替换内容中的PRICE为"<PRICE>"
        content = self.content_repatter6.sub("<PRICE>", content)
        # 使用content_trans1对content进行翻译
        content = content.translate(self.content_trans1)
        # 循环直到content中不再包含"<BLOCK><BLOCK>"
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        # 返回清洗后的内容
        return content
    # 将空格替换为"<SP>"
    text = text.replace(" ", "<SP>")
    # 将全角空格替换为"<SP>"
    text = text.replace("　", "<SP>")
    # 将换行符替换为"<BR>"
    text = text.replace("\r\n", "<BR>")
    text = text.replace("\n", "<BR>")
    text = text.replace("\r", "<BR>")
    # 将制表符替换为"<TAB>"
    text = text.replace("\t", "<TAB>")
    # 将特殊符号替换为对应的字符
    text = text.replace("—", "ー")
    text = text.replace("−", "ー")
    
    # 遍历表情字典，将文本中的表情符号替换为对应的字符
    for k, v in self.emoji["emoji"].items():
        if k in text:
            text = text.replace(k, v)
    
    # 如果需要清洗文本，则调用clean_text方法进行清洗
    if clean:
        text = self.clean_text(text)

    # 检查是否为特殊符号
    def check_simbol(x):
        # 将字符编码为字节流
        e = x.encode()
        if len(x) == 1 and len(e) == 2:
            c = (int(e[0]) << 8) + int(e[1])
            if (
                (c >= 0xC2A1 and c <= 0xC2BF)
                or (c >= 0xC780 and c <= 0xC783)
                or (c >= 0xCAB9 and c <= 0xCBBF)
                or (c >= 0xCC80 and c <= 0xCDA2)
            ):
                return True
        return False

    # 检查是否为特殊符号
    def checku2e(x):
        # 将字符编码为字节流
        e = x.encode()
        if len(x) == 1 and len(e) == 3:
            c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
            if c >= 0xE28080 and c <= 0xE2B07F:
                return True
        return False

    pos = 0
    result = []
    # 循环处理文本
    while pos < len(text):
        # 设置结束位置
        end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
        candidates = []  # 存储候选词的列表 (token_id, token, pos)
        # 从结束位置向前遍历
        for e in range(end, pos, -1):
            wd = text[pos:e]
            # 如果词在词汇表中，则加入候选列表
            if wd in self.vocab:
                if wd[0] == "<" and len(wd) > 2:
                    candidates = [(self.vocab[wd], wd, e)]
                    break
                else:
                    candidates.append((self.vocab[wd], wd, e))
        if len(candidates) > 0:
            # 选择最小的token_id
            _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
            result.append(wd)
            pos = e
        else:
            end = pos + 1
            wd = text[pos:end]
            # 检查是否为特殊符号
            if check_simbol(wd):
                result.append("<KIGOU>")
            # 检查是否为特殊符号
            elif checku2e(wd):
                result.append("<U2000U2BFF>")
            else:
                # 将字符编码为utf-8字节流
                for i in wd.encode("utf-8"):
                    result.append("<|byte%d|>" % i)
            pos = end
    return result
    # 将给定索引转换为对应的标记
    def convert_id_to_token(self, index, breakline="\n"):
        # 初始化空列表用于存储单词和字节标记
        words = []
        byte_tokens = []
        # 获取索引对应的单词
        word = self.ids_to_tokens[index][0]
        # 检查是否为字节标记
        if word[:6] == "<|byte" and word[-2:] == "|>":
            byte_tokens.append(int(word[6:-2]))
        else:
            # 如果存在字节标记，则将其解码为字符串并添加到单词列表中
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                byte_tokens = []
            # 根据不同的特殊标记进行处理
            if word[:7] == "<|emoji" and word[-2:] == "|>":
                words.append(self.emoji["emoji_inv"][word])
            elif word == "<SP>":
                words.append(" ")
            elif word == "<BR>":
                words.append(breakline)
            elif word == "<TAB>":
                words.append("\t")
            elif word == "<BLOCK>":
                words.append("▀")
            elif word == "<KIGOU>":
                words.append("ǀ")
            elif word == "<U2000U2BFF>":
                words.append("‖")
            else:
                words.append(word)
        # 如果存在未处理的字节标记，则解码为字符串并添加到单词列表中
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        # 将单词列表连接成文本并返回
        text = "".join(words)
        return text
```