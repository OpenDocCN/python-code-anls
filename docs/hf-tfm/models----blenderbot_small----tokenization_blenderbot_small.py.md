# `.\transformers\models\blenderbot_small\tokenization_blenderbot_small.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可证
"""Tokenization class for BlenderbotSmall."""

# 导入必要的模块
import json
import os
from typing import Dict, List, Optional, Tuple

# 导入 regex 作为 re
import regex as re

# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ...tokenization_utils import PreTrainedTokenizer
# 导入 logging 工具
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇表文件名字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json"
    },
    "merges_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt"
    },
    "tokenizer_config_file": {
        "facebook/blenderbot_small-90M": (
            "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json"
        )
    },
}

# 定义预训练位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot_small-90M": 512}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 返回一个单词中的符号对集合
    # 单词表示为符号的元组（符号是可变长度的字符串）
    pairs = set()
    prev_char = word[0]
    # 遍历单词的每个字符
    for char in word[1:]:
        # 添加符号对到集合中
        pairs.add((prev_char, char))
        prev_char = char

    # 将符号对集合转为集合（去除重复的对）
    pairs = set(pairs)
    return pairs


class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    the superclass for more information regarding methods.

    """
    Args:
        vocab_file (`str`):
            词汇表文件路径。
        merges_file (`str`):
            合并文件路径。
        bos_token (`str`, *optional*, defaults to `"__start__"`):
            句子开始标记。
        eos_token (`str`, *optional*, defaults to `"__end__"`):
            句子结束标记。
        unk_token (`str`, *optional*, defaults to `"__unk__"`):
            未知标记。词汇表中不存在的标记将被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"__null__"`):
            填充标记，在批处理具有不同长度的序列时使用。
        kwargs (*optional*):
            传递给 [`PreTrainedTokenizer`] 的额外关键字参数。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="__start__",
        eos_token="__end__",
        unk_token="__unk__",
        pad_token="__null__",
        **kwargs,
    ):
        # 用 UTF-8 编码打开词汇表文件，并加载为字典作为编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 解码器是编码器的反转形式
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 用 UTF-8 编码打开合并文件，并解析为元组列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        # 创建 BPE 标记的等级字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 创建缓存字典
        self.cache = {}
        # 调用父类的初始化方法，传递未知标记、句子开始、句子结束和填充标记等参数
        super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        # 返回词汇表大小
        return len(self.encoder)

    def get_vocab(self) -> Dict:
        # 返回词汇表和添加标记编码器的组合字典
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token: str) -> str:
        # 如果 token 已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 使用正则表达式将特殊字符前后加上空格，例如将".", ",", "!"等加上空格
        token = re.sub("([.,!?()])", r" \1", token)
        # 使用正则表达式在单引号前后加上空格
        token = re.sub("(')", r" \1 ", token)
        # 使用正则表达式将连续的空格替换成一个空格
        token = re.sub(r"\s{2,}", " ", token)
        # 如果 token 中包含换行符，将其替换为 "__newln__"
        if "\n" in token:
            token = token.replace("\n", " __newln__")

        # 将 token 按空格拆分成单词
        tokens = token.split(" ")
        # 存储拆分后的单词
        words = []
        # 遍历每个单词
        for token in tokens:
            # 如果单词长度为 0，跳过
            if not len(token):
                continue

            # 将单词转换为小写形式
            token = token.lower()
            # 将单词转换为字符元组
            word = tuple(token)
            # 将单词最后一个字符与 "</w>" 组成元组，表示单词结束
            word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            # 获取单词的字符对
            pairs = get_pairs(word)

            # 如果不存在字符对，直接将单词添加到结果中
            if not pairs:
                words.append(token)
                continue

            # 循环处理字符对，直到无法继续合并
            while True:
                # 找到频率最低的字符对
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                # 如果字符对不在字符对频率字典中，跳出循环
                if bigram not in self.bpe_ranks:
                    break
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
                new_word = tuple(new_word)
                word = new_word
                # 如果单词长度为 1，跳出循环
                if len(word) == 1:
                    break
                else:
                    pairs = get_pairs(word)
            # 将单词转换为 BPE 形式，并移除末尾的 "@@"
            word = "@@ ".join(word)
            word = word[:-4]

            # 将 token 与对应的 BPE 形式缓存起来
            self.cache[token] = word
            # 将 BPE 形式的单词添加到结果中
            words.append(word)
        # 将结果单词列表转换为字符串并返回
        return " ".join(words)

    def _tokenize(self, text: str) -> List[str]:
        """Split a string into tokens using BPE."""
        # 初始化拆分后的 token 列表
        split_tokens = []

        # 使用正则表达式将文本按非空白字符拆分成单词
        words = re.findall(r"\S+\n?", text)

        # 遍历每个单词，并使用 BPE 将其拆分成子词，将子词添加到结果列表中
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        # 返回拆分后的 token 列表
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token to an id using the vocab."""
        # 将 token 转换为小写形式，并从词汇表中获取对应的 id，如果不存在则返回未知标记的 id
        token = token.lower()
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引转换为对应的 token，如果不存在则返回未知标记
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens in a single string."""
        # 将 token 列表连接成字符串，并移除 BPE 标记 "@@"
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        # 返回连接后的字符串
        return out_string
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 拼接合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引
        index = 0
        # 写入合并信息到文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历BPE标记和标记索引，按索引排序
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 如果索引不连续，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 写入BPE标记到文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    @property
    # 从transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template复制而来
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 记录警告信息，提醒没有定义聊天模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```