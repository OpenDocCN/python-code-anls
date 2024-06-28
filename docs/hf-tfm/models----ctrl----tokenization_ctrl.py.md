# `.\models\ctrl\tokenization_ctrl.py`

```
# coding=utf-8
# 设置文件的字符编码为UTF-8，确保可以正确处理中文等特殊字符
# Copyright 2018 Salesforce and The HuggingFace Inc. team.
# 版权声明，声明代码的版权归属
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可协议授权使用本代码
# you may not use this file except in compliance with the License.
# 除非符合许可协议，否则不得使用此文件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 可以在上述链接获取许可协议的副本
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 本代码基于 "AS IS" 分发，无论明示还是暗示，不提供任何担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可协议以了解具体的使用条款和限制条件
"""Tokenization classes for Salesforce CTRL."""
# 用于 Salesforce CTRL 模型的分词类

import json
# 导入json模块，用于处理JSON格式数据
import os
# 导入os模块，用于处理操作系统相关的功能
from typing import Optional, Tuple
# 导入必要的类型提示模块，用于声明函数的参数和返回值类型

import regex as re
# 导入regex模块，用于处理正则表达式

from ...tokenization_utils import PreTrainedTokenizer
# 从父目录的tokenization_utils模块中导入PreTrainedTokenizer类
from ...utils import logging
# 从父目录的utils模块中导入logging工具

logger = logging.get_logger(__name__)
# 使用logging模块获取当前模块的logger对象

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}
# 定义词汇文件和合并文件的名称映射，用于CTRL模型的加载

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"Salesforce/ctrl": "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-vocab.json"},
    "merges_file": {"Salesforce/ctrl": "https://raw.githubusercontent.com/salesforce/ctrl/master/ctrl-merges.txt"},
}
# 预训练模型的词汇文件和合并文件的URL映射，用于CTRL模型的加载

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/ctrl": 256,
}
# 预训练模型位置嵌入的尺寸映射，用于CTRL模型的加载

CONTROL_CODES = {
    "Pregnancy": 168629,
    "Christianity": 7675,
    "Explain": 106423,
    "Fitness": 63440,
    "Saving": 63163,
    "Ask": 27171,
    "Ass": 95985,
    "Joke": 163509,
    "Questions": 45622,
    "Thoughts": 49605,
    "Retail": 52342,
    "Feminism": 164338,
    "Writing": 11992,
    "Atheism": 192263,
    "Netflix": 48616,
    "Computing": 39639,
    "Opinion": 43213,
    "Alone": 44967,
    "Funny": 58917,
    "Gaming": 40358,
    "Human": 4088,
    "India": 1331,
    "Joker": 77138,
    "Diet": 36206,
    "Legal": 11859,
    "Norman": 4939,
    "Tip": 72689,
    "Weight": 52343,
    "Movies": 46273,
    "Running": 23425,
    "Science": 2090,
    "Horror": 37793,
    "Confession": 60572,
    "Finance": 12250,
    "Politics": 16360,
    "Scary": 191985,
    "Support": 12654,
    "Technologies": 32516,
    "Teenage": 66160,
    "Event": 32769,
    "Learned": 67460,
    "Notion": 182770,
    "Wikipedia": 37583,
    "Books": 6665,
    "Extract": 76050,
    "Confessions": 102701,
    "Conspiracy": 75932,
    "Links": 63674,
    "Narcissus": 150425,
    "Relationship": 54766,
    "Relationships": 134796,
    "Reviews": 41671,
    "News": 4256,
    "Translation": 26820,
    "multilingual": 128406,
}
# 控制代码映射，将特定的控制名称映射到其对应的数字代码

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 返回单词中所有符号对的集合
    # 单词被表示为符号（符号是长度可变的字符串）的元组
    pairs = set()
    prev_char = word[0]
    # 初始化前一个字符为单词的第一个字符
    for char in word[1:]:
        # 遍历单词中的每一个字符（从第二个字符开始）
        pairs.add((prev_char, char))
        # 将当前字符和前一个字符组成的符号对添加到集合中
        prev_char = char
        # 更新前一个字符为当前字符

    pairs = set(pairs)
    # 转换为集合类型并返回
    return pairs


class CTRLTokenizer(PreTrainedTokenizer):
    """
    Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.
    构造一个CTRL分词器，基于字节对编码（Byte-Pair-Encoding）。
    """
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    # 定义类级别的常量和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    control_codes = CONTROL_CODES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        # 从给定的词汇文件中加载编码器（字典）
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器，是编码器的反转映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 从给定的合并文件中读取 BPE（Byte-Pair Encoding）合并操作
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        # 创建 BPE 合并操作到排名的映射
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存，用于存储已处理过的 BPE 操作结果
        self.cache = {}
        # 调用父类的初始化方法，传入未知标记和额外的关键字参数
        super().__init__(unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self):
        # 返回编码器中的词汇大小（词汇表大小）
        return len(self.encoder)

    def get_vocab(self):
        # 返回编码器和添加的特殊标记编码器合并后的字典
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        # 如果缓存中已经存在对应的 BPE 结果，则直接返回
        if token in self.cache:
            return self.cache[token]
        # 将单词转换为字符元组，并添加结束符</w>，以进行 BPE 操作
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        # 获取单词中的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回原始标记
        if not pairs:
            return token

        while True:
            # 找到优先级最高的字符对进行合并
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找不到该字符对的合并操作，停止循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 执行 BPE 合并操作
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果单词长度为1，停止合并操作
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将合并结果转换为 BPE 格式的标记
        word = "@@ ".join(word)
        word = word[:-4]
        # 将处理过的结果存入缓存并返回
        self.cache[token] = word
        return word
    # 将输入的文本按非空白字符分割成单词列表
    def _tokenize(self, text):
        split_tokens = []

        words = re.findall(r"\S+\n?", text)

        # 遍历每个单词，并应用BPE编码器将每个单词拆分成子词，加入到split_tokens列表中
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    # 根据词汇表将token转换为对应的ID，如果token不在词汇表中，则使用unk_token对应的ID
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 根据词汇表将ID转换为对应的token，如果ID不在词汇表中，则使用unk_token
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    # 将一系列token转换为单个字符串，去除特殊token标记"@@"并去除两端空格
    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    # 将词汇表保存到指定目录下的文件中，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            # 如果保存目录不存在，则记录错误信息并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建保存词汇表文件和BPE合并规则文件的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词汇表以JSON格式写入到vocab_file
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE合并规则写入到merge_file
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果BPE合并索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # decode方法被注释掉，可能用于将token_ids解码为字符串，移除特殊标记和空格
    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)
```