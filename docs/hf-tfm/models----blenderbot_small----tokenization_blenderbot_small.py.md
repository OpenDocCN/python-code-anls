# `.\models\blenderbot_small\tokenization_blenderbot_small.py`

```py
# coding=utf-8
# 版权 2021 年 Facebook Inc. 和 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）进行许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据"原样"分发此软件
# 软件没有任何形式的担保或条件，无论是明示的还是暗示的。
# 有关特定语言的条款，请参阅许可证。
"""BlenderbotSmall 的分词类。"""

import json  # 导入 JSON 模块
import os  # 导入操作系统路径模块
from typing import Dict, List, Optional, Tuple  # 导入类型提示相关模块

import regex as re  # 导入正则表达式模块

from ...tokenization_utils import PreTrainedTokenizer  # 从 tokenization_utils 中导入 PreTrainedTokenizer 类
from ...utils import logging  # 从 utils 中导入 logging 模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
    "tokenizer_config_file": "tokenizer_config.json",  # 分词器配置文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json"
    },  # 预训练词汇文件映射
    "merges_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt"
    },  # 预训练合并文件映射
    "tokenizer_config_file": {
        "facebook/blenderbot_small-90M": (
            "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json"
        )
    },  # 预训练分词器配置文件映射
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot_small-90M": 512}  # 预训练位置嵌入尺寸映射


def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号的元组（符号是可变长度字符串）。
    """
    pairs = set()  # 创建空集合用于存储符号对
    prev_char = word[0]  # 获取单词的第一个符号
    for char in word[1:]:  # 遍历单词中的每个符号（除第一个符号外）
        pairs.add((prev_char, char))  # 将前一个符号和当前符号作为一个符号对加入集合
        prev_char = char  # 更新前一个符号为当前符号

    pairs = set(pairs)  # 去除重复的符号对（因为集合具有唯一性）
    return pairs  # 返回符号对集合


class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    基于 BPE（字节对编码）构建 Blenderbot-90M 分词器。

    此分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考
    超类以获取有关方法的更多信息。
    """
    pass  # 占位符，表示类的实现在此省略
    # 词汇文件的名称映射，用于指定特定模型的词汇文件
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，用于指定预训练模型的特定词汇文件
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入大小的映射，指定预训练模型的位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表，包含输入 ID 和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，用于创建一个新的分词器对象
    def __init__(
        self,
        vocab_file,  # 词汇文件的路径
        merges_file,  # 合并文件的路径
        bos_token="__start__",  # 句子开始标记，默认为 "__start__"
        eos_token="__end__",  # 句子结束标记，默认为 "__end__"
        unk_token="__unk__",  # 未知标记，用于词汇表中不存在的词，默认为 "__unk__"
        pad_token="__null__",  # 填充标记，用于序列填充，默认为 "__null__"
        **kwargs,  # 其他可选关键字参数
    ):
        # 使用 utf-8 编码打开词汇文件，并加载到 self.encoder 中作为词汇表
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建反向映射，将编码转换为词汇
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 使用 utf-8 编码打开合并文件，并读取为字符串，按行拆分（去掉首尾空行）
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        # 将每行合并操作转换为元组，创建 BPE 合并的排名字典
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存字典
        self.cache = {}
        # 调用父类 PreTrainedTokenizer 的初始化方法，传递未知标记、句子开始结束标记、填充标记等参数
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)

    # vocab_size 属性，返回词汇表大小
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # get_vocab 方法，返回包含 encoder 和 added_tokens_encoder 的词汇表字典
    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token: str) -> str:
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]

        # 根据正则表达式替换标点符号等符号
        token = re.sub("([.,!?()])", r" \1", token)
        token = re.sub("(')", r" \1 ", token)
        token = re.sub(r"\s{2,}", " ", token)

        # 如果 token 中包含换行符，则用 "__newln__" 替换换行符
        if "\n" in token:
            token = token.replace("\n", " __newln__")

        # 将 token 按空格分割成 tokens
        tokens = token.split(" ")
        words = []

        # 遍历每个 token
        for token in tokens:
            if not len(token):
                continue

            # 将 token 转换为小写
            token = token.lower()

            # 将 token 转换为字符元组
            word = tuple(token)

            # 将字符元组的最后一个字符和 "</w>" 组合成新的元组
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])

            # 获取字符元组的所有可能的 bigram 组合
            pairs = get_pairs(word)

            # 如果没有 bigram 组合，则直接将 token 添加到结果中
            if not pairs:
                words.append(token)
                continue

            # 处理包含 bigram 组合的情况
            while True:
                # 找到在当前 bigram 中排名最小的 bigram
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

                # 如果找不到 bigram 的排名，则跳出循环
                if bigram not in self.bpe_ranks:
                    break

                # 拆分当前字符元组，并根据找到的 bigram 合并字符
                first, second = bigram
                new_word = []
                i = 0

                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except ValueError:
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                # 更新字符元组为新的字符组合
                new_word = tuple(new_word)
                word = new_word

                # 如果字符元组长度为 1，则跳出循环
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)

            # 将处理后的字符元组转换为字符串，并去除末尾的 "@@" 标记
            word = "@@ ".join(word)
            word = word[:-4]

            # 将 token 及其对应的处理结果存入缓存中
            self.cache[token] = word
            words.append(word)

        # 将处理后的 tokens 组合成一个字符串并返回
        return " ".join(words)

    def _tokenize(self, text: str) -> List[str]:
        """Split a string into tokens using BPE."""
        split_tokens = []

        # 使用正则表达式找出所有非空白字符序列，包括可能的换行符
        words = re.findall(r"\S+\n?", text)

        # 遍历每个找到的 token，并将其通过 BPE 分词加入到 split_tokens 中
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))

        # 返回分词后的结果列表
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        # 将 token 转换为小写，并通过词汇表获取对应的 id，如果不存在则返回未知 token 的 id
        token = token.lower()
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引转换为对应的 token，如果不存在则返回未知 token
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string."""
        # 将 tokens 列表连接成一个字符串，并移除 "@@" 标记，然后去除首尾空格并返回结果
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 根据给定的前缀（如果有的话），构建词汇表文件的完整路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 根据给定的前缀（如果有的话），构建合并文件的完整路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入编码器（self.encoder）的 JSON 格式到词汇表文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入 BPE（字节对编码）的合并信息到合并文件中
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 写入文件版本信息
            writer.write("#version: 0.2\n")
            # 遍历并按照索引排序 BPE merge 信息，确保连续性并写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇表文件和合并文件的路径
        return vocab_file, merge_file

    @property
    # 从 transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template 复制
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 如果没有为该分词器定义聊天模板，则记录警告信息，并使用默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板，该模板用于在消息之间添加空格
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```