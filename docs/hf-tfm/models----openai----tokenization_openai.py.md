# `.\models\openai\tokenization_openai.py`

```
# 设定脚本的字符编码为 UTF-8
# 版权声明，使用 Apache License 2.0 开源许可协议
# 详细许可信息可访问 http://www.apache.org/licenses/LICENSE-2.0
# 如果不符合许可协议的要求，不得使用该文件
# 以下代码实现了 OpenAI GPT 的分词功能

# 导入所需的模块和库
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统功能的模块
import re  # 导入正则表达式模块
import unicodedata  # 导入处理 Unicode 数据的模块
from typing import Optional, Tuple  # 导入类型提示相关的功能

# 导入 tokenization_utils 模块中的函数和类
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 导入 logging 模块，用于日志记录
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称常量
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练模型使用的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/vocab.json"
    },
    "merges_file": {
        "openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/merges.txt"
    },
}

# 预训练模型使用的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-community/openai-gpt": 512,
}

# 以下是从 transformers.models.bert.tokenization_bert 中复制过来的函数
# 这个函数用于基本的空格分词处理
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到 tokens 列表
    tokens = text.split()
    # 返回分割后的 tokens 列表
    return tokens

# 以下是从 transformers.models.bert.tokenization_bert 中复制过来的类
# 这个类实现了基本的分词功能，包括标点符号分割、小写转换等
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    """
    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        """
        Initialize the Tokenizer with specified parameters.

        Args:
            do_lower_case (bool, optional, default=True): Whether or not to convert tokens to lowercase.
            never_split (Iterable, optional): Collection of tokens that should never be split during tokenization.
                                              Defaults to an empty list.
            tokenize_chinese_chars (bool, optional, default=True): Whether or not to tokenize Chinese characters.
            strip_accents (bool, optional): Whether or not to remove accents. If None, determined by `lowercase`.
            do_split_on_punc (bool, optional, default=True): Whether or not to split on punctuation marks.
        """
        # Initialize with default values or provided arguments
        if never_split is None:
            never_split = []
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
        # 使用 union() 方法将 `never_split` 参数与 `self.never_split` 属性合并，以确保不分割的词汇列表完整
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的空白和特殊字符
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果设定了 `tokenize_chinese_chars` 参数为真，则对文本进行中文字符的分词处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 标准化 Unicode 文本为 NFC 形式，确保不同的 Unicode 编码形式被视为相同的字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符进行分词，得到原始的 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个原始 token
        for token in orig_tokens:
            # 如果 token 不在 `never_split` 中，则可能对其进行小写处理和去重音符处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置了小写处理，则将 token 转换为小写
                    token = token.lower()
                    # 如果设置了去除重音符号，则对 token 进行去重音符处理
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果仅设置了去重音符号，则对 token 进行去重音符处理
                    token = self._run_strip_accents(token)
            # 将 token 进行标点符号的分割处理，并加入到分割后的 token 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 列表重新用空白字符连接为字符串，并进行最终的分词处理
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的 token 列表
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本标准化为 NFD 形式，以便去除重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取当前字符的 Unicode 分类信息
            cat = unicodedata.category(char)
            # 如果当前字符为重音符号，则跳过该字符，不加入到输出列表中
            if cat == "Mn":
                continue
            # 否则将当前字符加入到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串，并返回处理后的文本
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点处分割或者给定的文本在never_split中，则直接返回包含整个文本的列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 在输出列表中添加一个新的空列表，用于存储下一个词
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号
                if start_new_word:
                    # 添加一个空列表作为新词的起始
                    output.append([])
                start_new_word = False
                # 将当前字符添加到当前词的末尾
                output[-1].append(char)
            i += 1

        # 将每个词列表转换为字符串，并返回列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果是中文字符
            if self._is_chinese_char(cp):
                # 在中文字符的前后添加空格，并添加到输出列表中
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符连接成一个字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的码点是否是CJK字符的码点范围内
        # 这里参考了CJK统一表意文字的Unicode块范围
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符为无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 否则将字符添加到输出列表中
                output.append(char)
        # 将输出列表中的字符连接成一个字符串并返回
        return "".join(output)
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    # Initialize an empty set to store symbol pairs
    pairs = set()
    # Initialize the previous character as the first character in the word
    prev_char = word[0]
    # Iterate over each character in the word starting from the second character
    for char in word[1:]:
        # Add the pair of previous character and current character to the set
        pairs.add((prev_char, char))
        # Update the previous character to the current character for the next iteration
        prev_char = char
    # Return the set of symbol pairs
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization
    """
    # Replace em dashes, en dashes, horizontal bars, and ellipses with standard symbols
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    # Use regex to standardize certain punctuation marks with surrounding spaces
    text = re.sub(r"""(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""", r" \1 ", text)
    # Normalize line breaks to ensure consistent whitespace around them
    text = re.sub(r"\s*\n\s*", " \n ", text)
    # Replace multiple spaces and tabs with a single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip leading and trailing whitespace from the text
    return text.strip()


class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

    - lowercases all inputs,
    - uses `SpaCy` tokenizer and `ftfy` for pre-BPE tokenization if they are installed, fallback to BERT's
      `BasicTokenizer` if not.

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

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        try:
            # Attempt to import necessary libraries for tokenization and text fixing
            import ftfy
            from spacy.lang.en import English

            # Use SpaCy tokenizer and ftfy text fixing
            _nlp = English()
            self.nlp = _nlp.tokenizer
            self.fix_text = ftfy.fix_text
        except ImportError:
            # Warn if libraries are not available and fallback to BERT's BasicTokenizer
            logger.warning("ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.")
            self.nlp = BasicTokenizer(do_lower_case=True)
            self.fix_text = None

        # Load vocabulary and merges files into the tokenizer
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # Create a reverse dictionary for decoding IDs to tokens
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        # Create a dictionary of BPE merges with their respective ranks
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # Initialize a cache for storing tokenization results
        self.cache = {}

        # Initialize the tokenizer using the superclass method
        super().__init__(unk_token=unk_token, **kwargs)
    # 返回 True，表示执行小写转换
    def do_lower_case(self):
        return True

    # 返回词汇表大小，即编码器中的条目数
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 返回包含编码器和额外令牌编码器的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 对给定的单词进行 BPE（Byte Pair Encoding）处理
    def bpe(self, token):
        # 如果缓存中已存在该单词的处理结果，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]

        # 将单词转换为 BPE 处理过的形式
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        # 如果没有找到任何对，则返回带结束符的原始单词
        if not pairs:
            return token + "</w>"

        # 开始迭代处理单词中的 bigram
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0

            # 遍历单词，处理 bigram
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

            # 更新单词，继续处理直到不再有 bigram
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # 将处理后的单词转换为字符串形式，并处理特殊情况
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"

        # 将处理结果添加到缓存中并返回
        self.cache[token] = word
        return word

    # 对文本进行标记化处理，返回标记化后的字符串列表
    def _tokenize(self, text):
        split_tokens = []

        # 如果未指定修正文本处理器，则使用 BERT 的基础标记器进行处理
        if self.fix_text is None:
            text = self.nlp.tokenize(text)
            for token in text:
                split_tokens.extend(list(self.bpe(token).split(" ")))
        else:
            # 使用 SpaCy 和 ftfy 进行原始的标记化处理（OpenAI GPT 的标记化过程）
            text = self.nlp(text_standardize(self.fix_text(text)))
            for token in text:
                split_tokens.extend(list(self.bpe(token.text.lower()).split(" ")))

        return split_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token（BPE 格式），如果未找到，则返回 unk_token
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    # 将一系列 token（字符串）转换为单一字符串
    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string
    # 定义一个保存词汇表的方法，接收一个保存目录路径和可选的文件名前缀作为参数，返回一个元组类型的结果
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件路径，如果有文件名前缀则添加到文件名中
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 构建合并文件路径，如果有文件名前缀则添加到文件名中
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词典内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引值
        index = 0
        # 打开合并文件，以 UTF-8 编码写入内容
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 在文件开头写入版本信息
            writer.write("#version: 0.2\n")
            # 遍历并排序 BPE 标记及其索引，按照索引值升序排列
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 如果当前索引值不等于期望的索引值，则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 将 BPE 标记以空格分隔并写入文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇表文件路径和合并文件路径组成的元组
        return vocab_file, merge_file
```