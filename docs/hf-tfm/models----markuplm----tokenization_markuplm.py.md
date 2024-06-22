# `.\transformers\models\markuplm\tokenization_markuplm.py`

```py
# 设置编码格式为 UTF-8，并声明版权信息
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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

"""Tokenization class for MarkupLM."""

# 导入所需模块
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

# 导入 regex 库，命名为 re
import regex as re

# 从文件工具中导入必要的函数和对象
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
# 从基础的 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 从基础的 tokenization_utils_base 模块中导入所需的类和常量
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
# 导入日志模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义用于模型词汇表的文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/vocab.json",
        "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/vocab.json",
    },
    "merges_file": {
        "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/merges.txt",
        "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/merges.txt",
    },
}

# 定义预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/markuplm-base": 512,
    "microsoft/markuplm-large": 512,
}

# 下面是字节到 Unicode 字符的转换函数的定义，使用 lru_cache 进行缓存
# 函数返回 utf-8 字节的列表及其与 Unicode 字符的映射
# 该函数特意避免了将空白字符和控制字符映射到 Unicode 字符，因为 BPE 编码会出错
# 可逆的 BPE 编码适用于 Unicode 字符串，因此如果要避免 UNK（未知）标记，您需要在词汇表中具有大量的 Unicode 字符
# 当您拥有大约 100 亿个标记数据集时，您最终需要大约 5K 个 Unicode 字符才能获得良好的覆盖率
# 这占了您正常的词汇表的相当大比例，例如，32K BPE 词汇表
# 为了避免这种情况，我们想要在 utf-8 字节和 Unicode 字符之间建立查找表

@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 定义用于获取单词对的函数
# 该函数将一个单词作为输入，并返回单词中的所有可能的 BPE 对
# BPE 对是指 BPE 算法中的基本元素，表示两个连续字符在词汇表中合并的方式

def get_pairs(word):
    # 返回一个单词中的符号对集合。单词表示为符号元组（符号为可变长度的字符串）。
    def symbol_pairs_in_word(word):
        # 初始化符号对集合
        pairs = set()
        # 初始化前一个符号为单词的第一个符号
        prev_char = word[0]
        # 遍历单词中的每个符号（除了第一个符号）
        for char in word[1:]:
            # 将前一个符号与当前符号组成一个符号对，并添加到符号对集合中
            pairs.add((prev_char, char))
            # 更新前一个符号为当前符号，为下一次循环做准备
            prev_char = char
        # 返回符号对集合
        return pairs
# MarkupLMTokenizer 类是一个基于 Byte-Pair-Encoding (BPE) 的预训练分词器
class MarkupLMTokenizer(PreTrainedTokenizer):
    r"""
    # 该类继承自 PreTrainedTokenizer，用于将 HTML 字符串转换为以下 token 级别的输出:
    #   - input_ids
    #   - attention_mask
    #   - token_type_ids
    #   - xpath_tags_seq
    #   - xpath_tags_seq
    # 用户可以参考父类 PreTrainedTokenizer 中的主要方法
    Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE). [`MarkupLMTokenizer`] can be used to
    turn HTML strings into to token-level `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and
    `xpath_tags_seq`. This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径。
        merges_file (`str`):
            # 合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            # 在将字节解码为 UTF-8 时采用的策略。参见官方文档关于 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) 的更多信息。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 在预训练过程中用作序列开始标记的 token。也可以用作序列分类器的 token。

            # 提示

            # 在使用特殊 token 构建序列时，不是用这个 token 作为序列开始的 token。而是用 `cls_token`。

            # 提示结束

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 在序列的结尾处的 token。

            # 提示

            # 在使用特殊 token 构建序列时，不是用这个 token 作为序列的结尾 token。而是用 `sep_token`。

            # 提示结束

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符 token，在从多个序列构建一个序列时使用，例如用于序列分类的两个序列，或用于文本和问题的问答场景。也用作使用特殊 token 构建序列时的最后一个 token。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 分类器 token，在进行序列分类时使用（对整个序列进行分类，而不是每个 token 的分类）。在使用特殊 token 构建序列时，是序列的第一个 token。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知 token，即词汇表中不存在的 token，无法转换为 ID，所以被设置为该 token。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 用于填充的 token，例如当批处理不同长度的序列时。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 用于掩码值的 token，在使用掩码语言建模进行模型训练时使用。模型将尝试预测该 token。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            # 是否在输入前添加一个初始空格。这样可以将开头的单词视为其他单词一样处理。 （RoBERTa 分词器通过前一个空格检测单词的开头）。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 词汇表文件名的映射。
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练词汇表文件名的映射。
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练位置嵌入的最大模型输入大小。
    # 初始化分词器
        def __init__(
            self,
            vocab_file, # 词汇表文件路径
            merges_file, # 合并操作文件路径
            tags_dict, # 标签字典
            errors="replace", # 错误处理方式
            bos_token="<s>", # 句子开始标记
            eos_token="</s>", # 句子结束标记
            sep_token="</s>", # 句子分隔标记
            cls_token="<s>", # 分类标记
            unk_token="<unk>", # 未知标记
            pad_token="<pad>", # 填充标记
            mask_token="<mask>", # 掩码标记
            add_prefix_space=False, # 是否添加前导空格
            max_depth=50, # 最大XPath深度
            max_width=1000, # 最大XPath宽度
            pad_width=1001, # 填充XPath宽度
            pad_token_label=-100, # 填充标签
            only_label_first_subword=True, # 是否仅标记第一个子词
            **kwargs,
        ):
            # ...
    
        # 获取XPath序列
        def get_xpath_seq(self, xpath):
            """
            给定一个节点的XPath表达式(如 "/html/body/div/li[1]/div/span[2]")，
            返回一个包含标签ID和下标的列表，考虑最大深度。
            """
            xpath_tags_list = [] # 标签ID列表
            xpath_subs_list = [] # 下标列表
    
            xpath_units = xpath.split("/") # 按"/"分割XPath表达式
            for unit in xpath_units:
                if not unit.strip():
                    continue
                name_subs = unit.strip().split("[") # 分离标签名和下标
                tag_name = name_subs[0] # 获取标签名
                sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1]) # 获取下标
                xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id)) # 将标签ID加入列表
                xpath_subs_list.append(min(self.max_width, sub)) # 将下标加入列表
    
            # 截断标签ID和下标列表，使其长度不超过最大深度
            xpath_tags_list = xpath_tags_list[: self.max_depth]
            xpath_subs_list = xpath_subs_list[: self.max_depth]
            # 使用填充标记补齐列表长度
            xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
            xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))
    
            return xpath_tags_list, xpath_subs_list
    
        # 获取词汇表大小
        @property
        def vocab_size(self):
            return len(self.encoder)
    
        # 获取完整词汇表
        def get_vocab(self):
            vocab = self.encoder.copy() # 复制词汇表
            vocab.update(self.added_tokens_encoder) # 更新词汇表
            return vocab
    # 使用字节对编码算法分割给定的 token，并返回编码后的结果
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 获取 token 组成的所有字节对
        pairs = get_pairs(word)

        # 如果没有字节对，则直接返回 token
        if not pairs:
            return token

        # 循环处理字节对
        while True:
            # 找出频率最低的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
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
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将编码后的 token 拼接成字符串形式
        word = " ".join(word)
        # 将结果存入缓存
        self.cache[token] = word
        return word

    # 将给定的文本进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # 使用正则表达式找到所有符合条件的 token
        for token in re.findall(self.pat, text):
            # 将每个 token 进行字节到 Unicode 字符串的映射，避免 BPE 中的控制标记（我们这里是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将对 token 进行 BPE 处理后的结果按空格切割，并加入到 bpe_tokens 中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将 token 转换成对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # ��� id 转换成对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列 tokens 转换成单个字符串形式
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        logger.warning(
            "MarkupLM now does not support generative tasks, decoding is experimental and subject to change."
        )
        # 将 tokens 拼接成单个字符串
        text = "".join(tokens)
        # 将 bytes 转换成 Unicode 字符串，使用 utf-8 编码并处理错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存路径是否是一个目录，如果不是则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 拼接合并文件名
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        
        # 保存词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 保存合并文件
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 遍历词汇表，写入合并文件
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表和合并文件名
        return vocab_file, merge_file

    # 为分词做准备
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 根据参数检查并处理输入文本
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    # 构建特殊标记的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # 以特殊标记拼接输入 IDs，形成 RoBERTa 模型的输入格式
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 构建带有特殊标记的 XPath 标记
    def build_xpath_tags_with_special_tokens(
        self, xpath_tags_0: List[int], xpath_tags_1: Optional[List[int]] = None
    ) -> List[int]:
        # 定义用于填充的特殊标记序列
        pad = [self.pad_xpath_tags_seq]
        # 如果 xpath_tags_1 为空，则返回填充标记 + xpath_tags_0 + 填充标记
        if len(xpath_tags_1) == 0:
            return pad + xpath_tags_0 + pad
        # 否则返回填充标记 + xpath_tags_0 + 填充标记 + xpath_tags_1 + 填充标记
        return pad + xpath_tags_0 + pad + xpath_tags_1 + pad

    def build_xpath_subs_with_special_tokens(
        self, xpath_subs_0: List[int], xpath_subs_1: Optional[List[int]] = None
    ) -> List[int]:
        # 定义用于填充的特殊标记序列
        pad = [self.pad_xpath_subs_seq]
        # 如果 xpath_subs_1 为空，则返回填充标记 + xpath_subs_0 + 填充标记
        if len(xpath_subs_1) == 0:
            return pad + xpath_subs_0 + pad
        # 否则返回填充标记 + xpath_subs_0 + 填充标记 + xpath_subs_1 + 填充标记
        return pad + xpath_subs_0 + pad + xpath_subs_1 + pad

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Args:
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # 如果已经存在特殊标记，则调用父类方法
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果 token_ids_1 为空，则返回带有特殊标记的 token_ids_0
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则返回带有特殊标记的 token_ids_0 和 token_ids_1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        # 定义用于分隔句子的特殊标记
        sep = [self.sep_token_id]
        # 定义用于表示句子开头的特殊标记
        cls = [self.cls_token_id]

        # 如果 token_ids_1 为空，则返回全零的列表，因为 RoBERTa 不使用 token type ids
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回全零的列表，因为 RoBERTa 不使用 token type ids
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个 __call__ 方法，用于文本编码处理
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],  # 接受文本输入，可以是单个文本或文本列表
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,  # 第二个文本输入，可选，可以是单个文本或文本列表
        xpaths: Union[List[List[int]], List[List[List[int]]]] = None,  # XPath路径，用于指定每个文本的标签
        node_labels: Optional[Union[List[int], List[List[int]]] = None,  # 节点标签，可选，用于指定每个文本的节点标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认禁用
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，可选
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的数据类型，如张量
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回文本长度
        verbose: bool = True,  # 是否启用详细模式
        **kwargs,  # 其他参数
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)  # 添加注解增强文档字符串
    # 定义一个 batch_encode_plus 方法，用于批处理文本编码
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],  # 批量文本或文本对输入，可以是文本列表、文本对列表、预分词文本列表
        is_pair: bool = None,  # 是否为文本对
        xpaths: Optional[List[List[List[int]]] = None,  # XPath路径，用于指定每个文本的标签
        node_labels: Optional[Union[List[int], List[List[int]]] = None,  # 节点标签，用于指定每个文本的节点标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认禁用
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，可选
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的数据类型，如张量
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回文本长度
        verbose: bool = True,  # 是否启用详细模式
        **kwargs,  # 其他参数
        # 返回批次编码的结果
        ) -> BatchEncoding:
        # 为了向后兼容 'truncation_strategy', 'pad_to_max_length'，获取填充和截断策略以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,  # 填充策略
            truncation=truncation,  # 截断策略
            max_length=max_length,  # 最大长度
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到最近的倍数
            verbose=verbose,  # 详细模式
            **kwargs,
        )

        # 对文本或文本对进行批量编码，并返回编码结果
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,  # 批量文本或文本对
            is_pair=is_pair,  # 是否为文本对
            xpaths=xpaths,  # XPath
            node_labels=node_labels,  # 节点标签
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度
            stride=stride,  # 步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到最近的倍数
            return_tensors=return_tensors,  # 返回张量
            return_token_type_ids=return_token_type_ids,  # 返回标记类型ID
            return_attention_mask=return_attention_mask,  # 返回注意力遮罩
            return_overflowing_tokens=return_overflowing_tokens,  # 返回溢出的标记
            return_special_tokens_mask=return_special_tokens_mask,  # 返回特殊标记的遮罩
            return_offsets_mapping=return_offsets_mapping,  # 返回偏移映射
            return_length=return_length,  # 返回长度
            verbose=verbose,  # 详细模式
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],  # 批量文本输入
            List[TextInputPair],  # 批量文本对输入
            List[PreTokenizedInput],  # 预分词输入列表
        ],
        is_pair: bool = None,  # 是否为文本对
        xpaths: Optional[List[List[List[int]]] = None,  # XPath
        node_labels: Optional[List[List[int]] = None,  # 节点标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到最近的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量
        return_token_type_ids: Optional[bool] = None,  # 返回标记类型ID
        return_attention_mask: Optional[bool] = None,  # 返回注意力遮罩
        return_overflowing_tokens: bool = False,  # 返回溢出的标记
        return_special_tokens_mask: bool = False,  # 返回特殊标记的遮罩
        return_offsets_mapping: bool = False,  # 返回偏移映射
        return_length: bool = False,  # 返回长度
        verbose: bool = True,  # 详细模式
        **kwargs,  # 其他参数
    # 定义一个方法，用于将输入文本或文本对批量准备为模型输入
    def __call__(
        self, 
        batch_text_or_text_pairs: Union[List[Union[str, Tuple[str, str]]], List[Prompt], List[List[Any]]], 
        return_offsets_mapping: bool = False, 
    ) -> BatchEncoding:
        # 如果请求返回偏移映射，则抛出NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用内部方法_batch_prepare_for_model来准备批量输入数据
        batch_outputs = self._batch_prepare_for_model(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            xpaths=xpaths,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 返回BatchEncoding对象
        return BatchEncoding(batch_outputs)

    # 添加端到端文档字符串和附加的参数文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,
        xpaths: Optional[List[List[int]]] = None,
        node_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    # 该函数用于将一个或多个输入序列准备为模型可以使用的格式
    def prepare_batch(
        self,
        batch_text_or_text_pairs: List[Union[str, Tuple[str, str]]],
        xpaths: List[str],
        node_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.LONGEST,
        truncation_strategy: TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_pair: bool = False,
        return_token_type_ids: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> BatchEncoding:
        """
        # 该函数的作用是:
        # 1. 为输入的序列添加特殊标记符号
        # 2. 如果序列长度超过最大长度,则根据策略截断
        # 3. 如果序列长度超过最大长度,使用指定的步长进行窗口滑动,并记录溢出的标记
        # 4. 为输入序列进行padding, 以便组成一个batch
    
        # 参数:
        # batch_text_or_text_pairs: 一个或多个输入序列,可以是单个序列或成对序列
        # xpaths: 与输入序列对应的XPath标签列表
        # node_labels: 可选的节点标签列表
        # 其他参数用于控制输入序列的处理方式,如是否添加特殊标记,截断策略,padding方式等
    
        # 返回值:
        # 一个BatchEncoding对象,包含了处理后的输入序列及其相关信息
        """
    
        # 初始化一个空的输出字典
        batch_outputs = {}
        
        # 遍历输入序列和XPath标签
        for idx, example in enumerate(zip(batch_text_or_text_pairs, xpaths)):
            batch_text_or_text_pair, xpaths_example = example
            
            # 对每个输入序列进行处理
            outputs = self.prepare_for_model(
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,
                batch_text_or_text_pair[1] if is_pair else None,
                xpaths_example,
                node_labels=node_labels[idx] if node_labels is not None else None,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )
    
            # 将每个输入序列的处理结果添加到输出字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
    
        # 对整个batch进行padding
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
    
        # 将处理后的batch封装成BatchEncoding对象
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
    
        return batch_outputs
    # 定义一个编码方法，将输入的文本或预分词输入编码为模型可接受的形式，并返回编码后的结果
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 定义输入的文本格式，可以是TextInput或PreTokenizedInput
        text_pair: Optional[PreTokenizedInput] = None,  # 定义文本对的格式，可选参数，默认为None
        xpaths: Optional[List[List[int]]] = None,  # 定义位置编码的列表格式，可选参数，默认为None
        node_labels: Optional[List[int]] = None,  # 定义节点标签的列表格式，可选参数，默认为None
        add_special_tokens: bool = True,  # 定义是否在输入中添加特殊标记，布尔值，默认为True
        padding: Union[bool, str, PaddingStrategy] = False,  # 定义填充策略，可以是布尔值、字符串或PaddingStrategy类型，默认为False
        truncation: Union[bool, str, TruncationStrategy] = None,  # 定义截断策略，可以是布尔值、字符串或TruncationStrategy类型，可选参数，默认为None
        max_length: Optional[int] = None,  # 定义输入的最大长度，可选参数，默认为None
        stride: int = 0,  # 定义步长，整数，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 定义填充到的倍数，可选参数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 定义返回的张量类型，可选参数，默认为None
        return_token_type_ids: Optional[bool] = None,  # 定义是否返回token类型的id，布尔值，可选参数，默认为None
        return_attention_mask: Optional[bool] = None,  # 定义是否返回注意力掩码，布尔值，可选参数，默认为None
        return_overflowing_tokens: bool = False,  # 定义是否返回溢出的token，布尔值，默认为False
        return_special_tokens_mask: bool = False,  # 定义是否返回特殊token的掩码，布尔值，默认为False
        return_offsets_mapping: bool = False,  # 定义是否返回偏移映射，布尔值，默认为False
        return_length: bool = False,  # 定义是否返回长度，布尔值，默认为False
        verbose: bool = True,  # 定义是否启用详细模式，布尔值，默认为True
        **kwargs,  # 接受任意额外的关键字参数
    ) -> List[int]:  # 指定返回类型为整数列表
        # 调用encode_plus方法，获取编码后的输入，并保存到encoded_inputs中
        encoded_inputs = self.encode_plus(
            text=text,
            text_pair=text_pair,
            xpaths=xpaths,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,  # 传递额外的关键字参数
        )
        # 返回编码后的输入中的input_ids
        return encoded_inputs["input_ids"]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个编码和更多方法，并添加相应的文档字符串
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 定义输入的文本格式，可以是TextInput或PreTokenizedInput
        text_pair: Optional[PreTokenizedInput] = None,  # 定义文本对的格式，可选参数，默认为None
        xpaths: Optional[List[List[int]]] = None,  # 定义位置编码的列表格式，可选参数，默认为None
        node_labels: Optional[List[int]] = None,  # 定义节点标签的列表格式，可选参数，默认为None
        add_special_tokens: bool = True,  # 定义是否在输入中添加特殊标记，布尔值，默认为True
        padding: Union[bool, str, PaddingStrategy] = False,  # 定义填充策略，可以是布尔值、字符串或PaddingStrategy类型，默认为False
        truncation: Union[bool, str, TruncationStrategy] = None,  # 定义截断策略，可以是布尔值、字符串或TruncationStrategy类型，可选参数，默认为None
        max_length: Optional[int] = None,  # 定义输入的最大长度，可选参数，默认为None
        stride: int = 0,  # 定义步长，整数，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 定义填充到的倍数，可选参数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 定义返回的张量类型，可选参数，默认为None
        return_token_type_ids: Optional[bool] = None,  # 定义是否返回token类型的id，布尔值，可选参数，默认为None
        return_attention_mask: Optional[bool] = None,  # 定义是否返回注意力掩码，布尔值，可选参数，默认为None
        return_overflowing_tokens: bool = False,  # 定义是否返回溢出的token，布尔值，默认为False
        return_special_tokens_mask: bool = False,  # 定义是否返回特殊token的掩码，布尔值，默认为False
        return_offsets_mapping: bool = False,  # 定义是否返回偏移映射，布尔值，默认为False
        return_length: bool = False,  # 定义是否返回长度，布尔值，默认为False
        verbose: bool = True,  # 定义是否启用详细模式，布尔值，默认为True
        **kwargs,  # 接受任意额外的关键字参数
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (nodes of a single example) or a
                list of list of strings (nodes of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，最大长度和其他参数，用于向后兼容
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用_encode_plus方法，为模型进行tokenize和准备输入
        return self._encode_plus(
            text=text,
            xpaths=xpaths,
            text_pair=text_pair,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        xpaths: Optional[List[List[int]]] = None,
        node_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    # 如果要求返回偏移量映射，则引发不实现的错误
    if return_offsets_mapping:
        raise NotImplementedError(
            "return_offset_mapping is not available when using Python tokenizers. "
            "To use this feature, change your tokenizer to one deriving from "
            "transformers.PreTrainedTokenizerFast. "
            "More information on available tokenizers at "
            "https://github.com/huggingface/transformers/pull/2674"
        )
    
    # 根据输入的各种参数准备模型输入数据
    return self.prepare_for_model(
        text=text,
        text_pair=text_pair,
        xpaths=xpaths,
        node_labels=node_labels,
        add_special_tokens=add_special_tokens,
        padding=padding_strategy.value,
        truncation=truncation_strategy.value,
        max_length=max_length,
        stride=stride,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors=return_tensors,
        prepend_batch_axis=True,
        return_attention_mask=return_attention_mask,
        return_token_type_ids=return_token_type_ids,
        return_overflowing_tokens=return_overflowing_tokens,
        return_special_tokens_mask=return_special_tokens_mask,
        return_length=return_length,
        verbose=verbose,
    )
    
    # 对输入的文本和配对文本进行预处理，并返回相应的 BatchEncoding 对象
    def prepare_for_model(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        xpaths: Optional[List[List[int]]] = None,
        node_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    )
    
    # 对输入序列进行截断，并返回截断后的序列及相关信息
    def truncate_sequences(
        self,
        ids: List[int],
        xpath_tags_seq: List[List[int]],
        xpath_subs_seq: List[List[int]],
        pair_ids: Optional[List[int]] = None,
        pair_xpath_tags_seq: Optional[List[List[int]]] = None,
        pair_xpath_subs_seq: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    )
    # 定义一个名为_pad的方法
    # 参数encoded_inputs表示要进行填充的编码输入，可以是字典或批量编码
    # 参数max_length表示最大长度，默认为None
    # 参数padding_strategy表示填充策略，默认为不进行填充
    # 参数pad_to_multiple_of表示填充到哪个倍数，默认为None
    # 参数return_attention_mask表示是否返回注意力掩码，默认为None
```