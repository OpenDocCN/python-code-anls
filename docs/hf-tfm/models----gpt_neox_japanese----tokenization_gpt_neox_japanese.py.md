# `.\models\gpt_neox_japanese\tokenization_gpt_neox_japanese.py`

```
# coding=utf-8
# 版权 2022 年 ABEJA, Inc. 和 The HuggingFace Inc. team. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"原样"提供的，
# 没有任何形式的明示或暗示担保或条件。
# 有关更多详细信息，请参阅许可证。
"""GPTNeoXJapanese 的标记化类。"""
import collections
import json
import os
import re
from typing import Optional, Tuple

import numpy as np

from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging

# 获取记录器实例
logger = logging.get_logger(__name__)

# 词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "emoji_file": "emoji.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/vocab.txt",
    },
    "emoji_file": {
        "abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/emoji.json",
    },
}

# 预训练位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "abeja/gpt-neox-japanese-2.7b": 2048,
}


def load_vocab_and_emoji(vocab_file, emoji_file):
    """加载词汇文件和表情文件到字典中。"""
    # 打开并加载表情文件为 JSON 格式
    with open(emoji_file, "r", encoding="utf-8") as f:
        emoji = json.loads(f.read())

    # 初始化字典
    vocab = collections.OrderedDict()
    raw_vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()

    # 打开并处理词汇文件
    with open(vocab_file, "r", encoding="utf-8") as f:
        token = f.readlines()

    # 格式化处理 token
    token = [[t.rstrip("\n")] if (t == "," or "," not in t) else t.rstrip("\n").split(",") for t in token]
    
    # 枚举 tokens
    for idx, b in enumerate(token):
        ids_to_tokens[idx] = b
        raw_vocab[",".join(b)] = idx
        for wd in b:
            vocab[wd] = idx

    return vocab, raw_vocab, ids_to_tokens, emoji


class GPTNeoXJapaneseTokenizer(PreTrainedTokenizer):
    """
    这个标记生成器继承自[`PreTrainedTokenizer`]，基于日本特殊的子词编码，该编码在此代码库中使用
    （https://github.com/tanreinama/Japanese-BPEEncoder_V2）。详细信息请参阅该代码库。
    日语词汇相对较大，并且单词之间没有分隔。此外，语言是由平假名、片假名和汉字组成，
    并且经常使用"1"和"①"等变体。为了应对这些情况，这个标记生成器具有以下功能：
    - 逐字子词分割，介于字节字符串和形态分析之间。
    """
    # 导入所需的GPTNeoXJapaneseTokenizer类
    from transformers import GPTNeoXJapaneseTokenizer
    
    # 定义GPTNeoXJapaneseTokenizer类，继承自Tokenizer类
    class GPTNeoXJapaneseTokenizer:
        # 类变量：定义词汇文件名列表
        vocab_files_names = VOCAB_FILES_NAMES
        # 类变量：定义预训练词汇文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 类变量：定义最大模型输入尺寸
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # 类变量：定义模型输入名称列表
        model_input_names = ["input_ids", "attention_mask"]
    
        # 初始化方法，接受多个参数
        def __init__(
            self,
            vocab_file,         # 词汇文件路径
            emoji_file,         # Emoji文件路径
            unk_token="<|endoftext|>",  # 未知标记的默认值
            pad_token="<|endoftext|>",  # 填充标记的默认值
            bos_token="<|startoftext|>",    # 序列开始标记的默认值
            eos_token="<|endoftext|>",  # 序列结束标记的默认值
            do_clean_text=False,    # 是否清理文本的标志，默认为False
            **kwargs,   # 其他关键字参数
        ):
            pass    # 初始化方法暂不做任何操作，保留扩展空间
    ):
        # 检查词汇文件是否存在，若不存在则抛出数值错误，指明路径，并建议从预训练模型加载
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 检查表情文件是否存在，若不存在则抛出数值错误，指明路径，并建议从预训练模型加载
        if not os.path.isfile(emoji_file):
            raise ValueError(
                f"Can't find an emoji file at path '{emoji_file}'. To load the emoji information from a Google"
                " pretrained model use `tokenizer = GPTNeoXJapaneseokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 设定是否进行文本清理的标志位
        self.do_clean_text = do_clean_text
        # 载入词汇和表情数据到相应的属性中
        self.vocab, self.raw_vocab, self.ids_to_tokens, self.emoji = load_vocab_and_emoji(vocab_file, emoji_file)
        # 初始化日语分词器，并传入必要的词汇、词汇到标记的映射、表情数据
        self.subword_tokenizer = SubWordJapaneseTokenizer(
            vocab=self.vocab, ids_to_tokens=self.ids_to_tokens, emoji=self.emoji
        )
        # 调用父类初始化方法，传入通用的参数及kwargs
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
        # 返回词汇表大小，即 raw_vocab 的长度
        return len(self.raw_vocab)

    def get_vocab(self):
        # 返回原始词汇表和添加的特殊标记编码的字典
        return dict(self.raw_vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # 使用子词日语分词器对文本进行分词处理，根据 do_clean_text 的设置决定是否进行文本清理
        return self.subword_tokenizer.tokenize(text, clean=self.do_clean_text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将给定的 token 转换成其对应的 id，若找不到则使用 unk_token 的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将给定的 index 转换成其对应的 token
        return self.subword_tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列的 token 转换成单个字符串，并去除首尾空格
        out_string = "".join(tokens).strip()
        return out_string

    @property
    def default_chat_template(self):
        """
        A simple chat template that just adds BOS/EOS tokens around messages while discarding role information.
        """
        # 若未定义聊天模板，则发出警告并使用默认模板，返回该模板字符串
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
    # 定义一个方法用于保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为0
        index = 0
        # 检查保存目录是否存在
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
            # 若保存目录不存在，则在文件名前加上前缀，构建词汇表文件路径
            vocab_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["vocab_file"]
            )
            # 构建表情符号文件路径
            emoji_file = (
                (filename_prefix + "-" if filename_prefix else "") + save_directory + VOCAB_FILES_NAMES["emoji_file"]
            )
        
        # 打开词汇表文件，使用utf-8编码方式写入数据
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的token索引和token内容
            for token_index, token in self.ids_to_tokens.items():
                # 检查索引是否连续，若不连续则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将token内容以逗号分隔写入文件，并换行
                writer.write(",".join(token) + "\n")
                # 更新索引
                index += 1
        
        # 打开表情符号文件，使用utf-8编码方式写入JSON格式的表情符号数据
        with open(emoji_file, "w", encoding="utf-8") as writer:
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
        self.vocab = vocab  # 词汇表，与 swe 相同
        self.ids_to_tokens = ids_to_tokens  # id 到 token 映射，与 bpe 相同
        self.emoji = emoji  # 表情符号
        self.maxlen = np.max([len(w) for w in self.vocab.keys()])  # 计算词汇表中最长词的长度
        # 定义多个正则表达式用于匹配特定的文本模式
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
        self.content_trans1 = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})  # 定义字符转换表，将一些特定字符替换为"<BLOCK>"

    def __len__(self):
        return len(self.ids_to_tokens)  # 返回 token 到 id 映射的长度
    # 清理文本内容，替换内容中的特定模式
    def clean_text(self, content):
        # 将内容中匹配到的 URL 替换为 "<URL>"
        content = self.content_repatter1.sub("<URL>", content)
        # 将内容中匹配到的 EMAIL 替换为 "<EMAIL>"
        content = self.content_repatter2.sub("<EMAIL>", content)
        # 将内容中匹配到的电话号码替换为 "<TEL>"
        content = self.content_repatter3.sub("<TEL>", content)
        # 将内容中匹配到的日期替换为 "<DATE>"
        content = self.content_repatter4.sub("<DATE>", content)
        # 再次将内容中匹配到的日期替换为 "<DATE>"
        content = self.content_repatter5.sub("<DATE>", content)
        # 将内容中匹配到的价格替换为 "<PRICE>"
        content = self.content_repatter6.sub("<PRICE>", content)
        # 使用指定的字符映射表进行字符转换
        content = content.translate(self.content_trans1)
        # 反复检查并替换连续的 "<BLOCK><BLOCK>" 为单个 "<BLOCK>"
        while "<BLOCK><BLOCK>" in content:
            content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
        # 返回清理后的内容
        return content
    # 定义一个方法，用于将文本进行分词处理，并可选地进行清理操作
    def tokenize(self, text, clean=False):
        # 将空格替换为特殊标记"<SP>"
        text = text.replace(" ", "<SP>")
        # 将全角空格替换为特殊标记"<SP>"
        text = text.replace("　", "<SP>")
        # 将Windows风格的换行符替换为特殊标记"<BR>"
        text = text.replace("\r\n", "<BR>")
        # 将Unix风格的换行符替换为特殊标记"<BR>"
        text = text.replace("\n", "<BR>")
        # 将老式Mac风格的换行符替换为特殊标记"<BR>"
        text = text.replace("\r", "<BR>")
        # 将制表符替换为特殊标记"<TAB>"
        text = text.replace("\t", "<TAB>")
        # 将特定字符替换为统一的字符"ー"
        text = text.replace("—", "ー")
        text = text.replace("−", "ー")
        
        # 替换文本中的表情符号为对应的Unicode字符串
        for k, v in self.emoji["emoji"].items():
            if k in text:
                text = text.replace(k, v)
        
        # 若clean参数为True，则调用clean_text方法清理文本
        if clean:
            text = self.clean_text(text)
        
        # 定义一个内部函数，用于检查是否为特定的符号字符
        def check_simbol(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 2:
                c = (int(e[0]) << 8) + int(e[1])
                # 判断是否符合日语、朝鲜语等特定范围内的字符编码
                if (
                    (c >= 0xC2A1 and c <= 0xC2BF)
                    or (c >= 0xC780 and c <= 0xC783)
                    or (c >= 0xCAB9 and c <= 0xCBBF)
                    or (c >= 0xCC80 and c <= 0xCDA2)
                ):
                    return True
            return False
        
        # 定义一个内部函数，用于检查是否为范围内的双字节字符
        def checku2e(x):
            e = x.encode()
            if len(x) == 1 and len(e) == 3:
                c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
                # 判断是否为Unicode范围内的字符
                if c >= 0xE28080 and c <= 0xE2B07F:
                    return True
            return False
        
        # 初始化位置变量
        pos = 0
        # 初始化结果列表
        result = []
        
        # 开始处理文本
        while pos < len(text):
            # 计算当前处理的结束位置
            end = min(len(text), pos + self.maxlen + 1) if text[pos] == "<" else pos + 3
            # 候选列表用于存储可能的token及其信息
            candidates = []  # (token_id, token, pos)
            
            # 从最大长度向当前位置遍历，找到最长的合法token
            for e in range(end, pos, -1):
                wd = text[pos:e]
                if wd in self.vocab:
                    if wd[0] == "<" and len(wd) > 2:
                        candidates = [(self.vocab[wd], wd, e)]
                        break
                    else:
                        candidates.append((self.vocab[wd], wd, e))
            
            # 若候选列表不为空，则选择token_id最小的token作为结果之一
            if len(candidates) > 0:
                _, wd, e = sorted(candidates, key=lambda x: x[0])[0]
                result.append(wd)
                pos = e
            else:
                # 若无合法token，则处理单个字符
                end = pos + 1
                wd = text[pos:end]
                # 检查是否为特定符号，若是则添加"<KIGOU>"标记
                if check_simbol(wd):
                    result.append("<KIGOU>")
                # 检查是否为范围内的双字节字符，若是则添加"<U2000U2BFF>"标记
                elif checku2e(wd):
                    result.append("<U2000U2BFF>")
                # 否则，按字节添加"<|byte%d|>"的标记
                else:
                    for i in wd.encode("utf-8"):
                        result.append("<|byte%d|>" % i)
                pos = end
        
        # 返回处理后的结果列表
        return result
    # 将给定的索引转换为对应的文本标记
    def convert_id_to_token(self, index, breakline="\n"):
        # 初始化一个空列表，用于存储最终的文本标记
        words = []
        # 初始化一个空列表，用于临时存储字节标记
        byte_tokens = []
        # 获取索引处的标记
        word = self.ids_to_tokens[index][0]
        
        # 检查是否是字节标记
        if word[:6] == "<|byte" and word[-2:] == "|>":
            # 提取字节标记的值并添加到字节标记列表中
            byte_tokens.append(int(word[6:-2]))
        else:
            # 如果之前有未处理的字节标记，则解码并添加到最终文本标记列表中
            if len(byte_tokens) > 0:
                words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
                byte_tokens = []
            
            # 根据特定标记进行处理
            if word[:7] == "<|emoji" and word[-2:] == "|>":
                # 如果是表情符号标记，则根据索引获取对应的表情符号并添加到文本标记列表中
                words.append(self.emoji["emoji_inv"][word])
            elif word == "<SP>":
                words.append(" ")  # 空格标记
            elif word == "<BR>":
                words.append(breakline)  # 换行符标记
            elif word == "<TAB>":
                words.append("\t")  # 制表符标记
            elif word == "<BLOCK>":
                words.append("▀")  # 方块字符标记
            elif word == "<KIGOU>":
                words.append("ǀ")  # 竖线符号标记
            elif word == "<U2000U2BFF>":
                words.append("‖")  # 双竖线符号标记
            else:
                words.append(word)  # 普通文本标记
        
        # 处理最后可能残留的字节标记并添加到文本标记列表中
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode("utf-8", errors="replace"))
        
        # 将所有文本标记连接成一个字符串
        text = "".join(words)
        # 返回转换后的文本字符串
        return text
```