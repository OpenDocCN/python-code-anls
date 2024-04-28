# `.\transformers\models\openai\tokenization_openai.py`

```
# 设置编码格式为 utf-8
# 版权声明：2018年由 Open AI 团队作者和 HuggingFace 公司团队发布
# 根据 Apache License, Version 2.0 可以在遵守许可证的前提下使用本文件
# 在以下网址可以获取许可证的拷贝
# http://www.apache.org/licenses/LICENSE-2.0
# 除非依据相关法律或经书面同意，否则仅以"现状"方式分发软件，没有任何明示或暗示的担保或条件。
# 详细信息可查看许可证
"""OpenAI GPT 的分词类别"""

# 导入模块
import json
import os
import re
import unicodedata
from typing import Optional, Tuple

# 导入其他相关的模块
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 词汇文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/vocab.json"},
    "merges_file": {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/merges.txt"},
}

# 预训练位置嵌入的尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-gpt": 512,
}

# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制过来的函数
def whitespace_tokenize(text):
    """在文本上执行基本的空格清理和分割"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制过来的类
class BasicTokenizer(object):
    """
    构建一个基本的 Tokenizer，用于运行基本的分词（标点符号分割、小写等）。

    参数:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            当分词的时候是否转换成小写。
        never_split (`Iterable`, *optional*):
            在分词过程中永远不会分割的符号集合。只在`do_basic_tokenize=True`时生效
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            当分词时是否要分割中文字符。对于日文应该禁用此功能
        strip_accents (`bool`, *optional*):
            是否去掉所有的音调符号。如果没有指定此选项，将会按照`lowercase`的值来决定（与原始的BERT一样）
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们想要跳过基本的标点符号分割，以便后面的分词可以获取单词的完整上下文，比如缩略词。
    """
    # 初始化 Tokenizer 类的实例
    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 为 None，则设置为一个空列表
        if never_split is None:
            never_split = []
        # 设置对象属性值
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 获取 never_split 和参数中的 never_split 的并集
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本数据
        text = self._clean_text(text)

        # 处理中文字符的分词
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 对文本进行 Unicode 规范化
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用边界空格分割文本，得到原始 token
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 在标点符号处分割 token
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 以空格拼接并再次用边界空格分割，得到最终输出 token
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 删除文本中的附加字符（accents）
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行 Unicode 规范化
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本字符，排除非组合字符（Mn）
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
```  
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割文本，或者指定了永远不分割的文本，则直接返回原始文本的列表形式
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历文本中的每个字符
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其放入一个单独的列表中，并设置标志以开始新的单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，则判断是否需要开始新的单词，然后将字符添加到当前单词中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 在任何中日韩（CJK）字符周围添加空格
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查 CP 是否是中日韩字符的代码点
        # 这里将中日韩字符定义为CJK Unicode块中的任何字符
        # 详情可参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        # 注意，CJK Unicode块并不包括所有的日语和韩语字符，例如现代韩文的“Hangul”字母是不同的块
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 对文本进行无效字符删除和空格清理
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
def get_pairs(word):
    """
    返回单词中的符号对集合。单词表示为符号元组（符号为可变长度字符串）。
    """
    # 初始化一个空集合，用于存储符号对
    pairs = set()
    # 初始化前一个字符为单词的第一个字符
    prev_char = word[0]
    # 遍历单词的每个字符，从第二个字符开始
    for char in word[1:]:
        # 将前一个字符和当前字符组成的符号对添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，用于下一轮迭代
        prev_char = char
    # 返回符号对集合
    return pairs


def text_standardize(text):
    """
    修复 spacy 分词器在书籍语料库中存在的一些问题，同时进行一些空格标准化
    """
    # 将一些特殊字符替换为标准形式
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    # 使用正则表达式进行文本标准化，替换一些特殊字符为它们的形式加上空格
    text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    # 将多余的空白字符替换为单个空格
    text = re.sub(r"\s*\n\s*", " \n ", text)
    # 将非空白字符前后的多余空格替换为单个空格
    text = re.sub(r"[^\S\n]+", " ", text)
    # 去除文本两端的空白字符
    return text.strip()


class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    构造一个 GPT 分词器。基于字节对编码，具有以下特点：

    - 将所有输入转换为小写，
    - 如果安装了 `SpaCy` 分词器和 `ftfy` 库，则使用它们进行 BPE 前的分词，否则使用 BERT 的 `BasicTokenizer`。

    这个分词器继承自 [`PreTrainedTokenizer`]，其中包含了大部分主要方法。用户应参考这个超类以获取更多关于这些方法的信息。

    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        unk_token (`str`, *optional*, 默认为 `"<unk>"`):
            未知标记。如果词汇表中不存在某个标记，则无法将其转换为 ID，而是将其设置为此标记。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        try:
            # 尝试导入 ftfy 和 SpaCy 库
            import ftfy
            from spacy.lang.en import English

            # 创建 SpaCy 分词器对象
            _nlp = English()
            self.nlp = _nlp.tokenizer
            self.fix_text = ftfy.fix_text
        except ImportError:
            # 如果导入失败，则使用 BERT 的 BasicTokenizer
            logger.warning("ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.")
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None

        # 从词汇表文件中加载编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器，将编码器的键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 从合并文件中加载字节对编码信息
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        # 创建字节对编码的排名字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存字典
        self.cache = {}

        # 调用父类构造函数进行初始化
        super().__init__(unk_token=unk_token, **kwargs)

    @property
    # 该函数用于判断是否进行小写转换
    def do_lower_case(self):
        # 返回 True，表示需要进行小写转换
        return True
    
    # 获取词汇表大小
    @property
    def vocab_size(self):
        # 返回词汇表大小，即 encoder 字典的长度
        return len(self.encoder)
    
    # 获取完整的词汇表
    def get_vocab(self):
        # 返回一个包含原有 encoder 词汇表和新增 token 编码的词汇表字典
        return dict(self.encoder, **self.added_tokens_encoder)
    
    # 将输入的 token 进行 BPE 编码
    def bpe(self, token):
        # 将最后一个字符后添加 "</w>" 标记，表示词的结束
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果该 token 在缓存中，直接返回缓存结果
        if token in self.cache:
            return self.cache[token]
        # 获取当前 word 中的所有 pair
        pairs = get_pairs(word)
    
        # 如果没有 pair，则直接返回 token 加上结束标记
        if not pairs:
            return token + "</w>"
    
        while True:
            # 找到当前 pairs 中频率最小的 pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该 pair 不在 bpe_ranks 中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 word 中的字符
            while i < len(word):
                try:
                    # 找到当前 word 中第一次出现 first 的位置
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到 first，则将剩余字符全部加入 new_word 并退出循环
                    new_word.extend(word[i:])
                    break
                else:
                    # 将 i 到 j 之间的字符加入 new_word
                    new_word.extend(word[i:j])
                    i = j
    
                # 如果 word[i] 是 first，且 i 还没到最后一个字符，且 word[i+1] 是 second
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 则将 first+second 作为一个新的字符加入 new_word
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将 word[i] 单独加入 new_word
                    new_word.append(word[i])
                    i += 1
            # 将 new_word 重新组成 word
            new_word = tuple(new_word)
            word = new_word
            # 如果 word 长度为 1，则退出循环
            if len(word) == 1:
                break
            else:
                # 否则继续获取 word 中的 pair
                pairs = get_pairs(word)
        # 将空格分隔的 word 拼接成字符串，如果是 "\n  </w>"，则改为 "\n</w>"
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果缓存并返回
        self.cache[token] = word
        return word
    
    # 对输入文本进行分词
    def _tokenize(self, text):
        # 初始化分词结果列表
        split_tokens = []
        # 如果 fix_text 为 None，则使用 BERT 的 BasicTokenizer 进行分词
        if self.fix_text is None:
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend(list(self.bpe(token).split(" ")))
        # 否则使用 SpaCy 和 ftfy 进行分词
        else:
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend(list(self.bpe(token.text.lower()).split(" ")))
        # 返回分词结果
        return split_tokens
    
    # 将 token 转换为 id
    def _convert_token_to_id(self, token):
        # 如果 token 在 encoder 中存在，返回其 id
        # 否则返回 unk_token 的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    # 将 id 转换为 token
    def _convert_id_to_token(self, index):
        # 如果 index 在 decoder 中存在，返回其对应的 token
        # 否则返回 unk_token
        return self.decoder.get(index, self.unk_token)
    
    # 将一序列 token 转换为字符串
    def convert_tokens_to_string(self, tokens):
        # 将 tokens 拼接成字符串，并去除结尾的 "</w>"
        out_string = "".join(tokens).replace("</w>", " ").strip()
        # 返回最终的字符串
        return out_string
    # 保存词汇表至指定目录，可选指定文件名前缀，默认返回文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 若不存在，记录错误信息并返回空值
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引为 0
        index = 0
        # 打开合并文件，写入版本信息
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历排序后的 BPE 标记及其索引
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查 BPE 合并索引是否连续
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    # 更新索引为当前索引
                    index = token_index
                # 将 BPE 标记以空格分隔写入文件
                writer.write(" ".join(bpe_tokens) + "\n")
                # 索引自增
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file
```