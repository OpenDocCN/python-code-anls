# `.\transformers\models\bertweet\tokenization_bertweet.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
import html  # 导入处理 HTML 的模块
import os  # 导入操作系统相关功能的模块
import re  # 导入正则表达式模块
from shutil import copyfile  # 导入文件复制功能
from typing import List, Optional, Tuple  # 导入类型提示相关功能

import regex  # 导入支持 Unicode 正则表达式的模块
from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器的基类
from ...utils import logging  # 导入日志记录工具

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",  # 词汇表文件名
    "merges_file": "bpe.codes",  # BPE 编码文件名
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/bertweet-base": "https://huggingface.co/vinai/bertweet-base/resolve/main/vocab.txt",  # 预训练模型的词汇表文件链接
    },
    "merges_file": {
        "vinai/bertweet-base": "https://huggingface.co/vinai/bertweet-base/resolve/main/bpe.codes",  # 预训练模型的 BPE 编码文件链接
    },
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vinai/bertweet-base": 128,  # vinai/bertweet-base 模型的位置嵌入尺寸为 128
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 返回单词中的符号对集合
    # 单词被表示为符号元组（符号是可变长度的字符串）
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class BertweetTokenizer(PreTrainedTokenizer):
    """
    Constructs a BERTweet tokenizer, using Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
```  
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径。
            Path to the vocabulary file.
        merges_file (`str`):
            # 合并文件的路径。
            Path to the merges file.
        normalization (`bool`, *optional*, defaults to `False`):
            # 是否应用标准化预处理。
            Whether or not to apply a normalization preprocess.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 在预训练期间用作序列开始标记的令牌。可以用作序列分类器令牌。

            <Tip>

            # 在使用特殊令牌构建序列时，这不是用作序列开始的令牌。使用的是`cls_token`。

            </Tip>

            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 序列结束标记。

            <Tip>

            # 在使用特殊令牌构建序列时，这不是用作序列结束的令牌。使用的是`sep_token`。

            </Tip>

            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符令牌，在从多个序列构建序列时使用，例如，用于序列分类或用于文本和问题的问题回答。也用作使用特殊令牌构建序列的最后一个令牌。
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 用于序列分类的分类器令牌（而不是每个令牌分类）。在使用特殊令牌构建序列时，它是序列的第一个令牌。
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知标记。不在词汇表中的标记无法转换为ID，并设置为此标记。
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 用于填充的标记，例如，当批处理不同长度的序列时。
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 用于屏蔽值的标记。这是在进行掩码语言建模训练时使用的标记。这是模型将尝试预测的标记。
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalization=False,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        # 尝试导入 emoji 模块中的 demojize 函数
        try:
            from emoji import demojize
            # 如果成功导入，将 demojize 函数赋值给实例属性 demojizer
            self.demojizer = demojize
        # 如果导入失败，捕获 ImportError 异常
        except ImportError:
            # 记录警告日志，说明 emoji 模块未安装，无法将表情符号或表情符号转换为文本
            logger.warning(
                "emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3"
                " install emoji==0.6.0"
            )
            # 将实例属性 demojizer 设置为 None
            self.demojizer = None

        # 初始化实例属性 vocab_file 和 merges_file
        self.vocab_file = vocab_file
        self.merges_file = merges_file

        # 初始化编码器字典，将特殊标记转换为对应的整数编码
        self.encoder = {}
        self.encoder[str(bos_token)] = 0
        self.encoder[str(pad_token)] = 1
        self.encoder[str(eos_token)] = 2
        self.encoder[str(unk_token)] = 3

        # 从文件中添加更多的编码
        self.add_from_file(vocab_file)

        # 初始化解码器字典，将整数编码转换为对应的特殊标记
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 从 merges 文件中读取 BPE 合并操作并构建 BPE 编码器
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        # 解析 merges 文件中的合并操作，构建 BPE 编码器
        merges = [tuple(merge.split()[:-1]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 初始化缓存
        self.cache = {}

        # 初始化文本规范化选项
        self.normalization = normalization
        # 初始化 TweetTokenizer 对象
        self.tweetPreprocessor = TweetTokenizer()
        # 初始化特殊标点符号字典
        self.special_puncts = {"’": "'", "…": "..."}

        # 调用父类的构造函数，初始化实例的其他属性
        super().__init__(
            normalization=normalization,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERTweet sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果只有一个序列，添加起始标记、序列和结束标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有两个序列，添加起始标记、第一个序列、两个结束标记、第二个序列和结束标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列中创建一个用于序列对分类任务的掩码。BERTweet 不使用标记类型 ID，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 如果 token 已经存在于缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 将 token 的最后一个字符添加结束标记 "</w>"，并转换为元组
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        # 获得 token 中所有字符的连续字符对
        pairs = get_pairs(word)

        # 如果没有连续字符对，则返回原始 token
        if not pairs:
            return token

        # 不断循环直到 token 中不存在更多连续字符对
        while True:
            # 找到出现次数最少的连续字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的字符对不在 BPE 词汇表中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            # 获取字符对的第一个字符和第二个字符
            first, second = bigram
            new_word = []
            i = 0
            # 循环 token 中的每个字符
            while i < len(word):
                try:
                    # 找到第一个字符的索引位置
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到第一个字符，则将剩余部分加入新的 token 中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将第一个字符之前的部分加入新的 token 中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符和下一个字符组成了连续字符对，则合并并加入新的 token 中
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符加入新的 token 中
                    new_word.append(word[i])
                    i += 1
            # 更新 token 为新的 token
            new_word = tuple(new_word)
            word = new_word
            # 如果新的 token 只有一个字符，则退出循环
            if len(word) == 1:
                break
            else:
                # 否则继续生成新的连续字符对
                pairs = get_pairs(word)
        # 将 token 中的所有字符用 '@@ ' 连接起来，并移除结束标记
        word = "@@ ".join(word)
        word = word[:-4]
        # 将 token 及其对应的结果存入缓存中
        self.cache[token] = word
        # 返回处理后的 token
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 如果启用了 Tweet normalization，则在进行 BPE 处理前进行 Tweet normalization
        if self.normalization:
            text = self.normalizeTweet(text)

        split_tokens = []
        # 使用正则表达式将文本分割成单词列表
        words = re.findall(r"\S+\n?", text)
        # 对每个单词进行 BPE 处理，并将结果添加到分词列表中
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        # 返回分词列表
        return split_tokens

    def normalizeTweet(self, tweet):
        """
        Normalize a raw Tweet
        """
        # 替换特殊标点符号
        for punct in self.special_puncts:
            tweet = tweet.replace(punct, self.special_puncts[punct])

        # 对 Tweet 进行分词
        tokens = self.tweetPreprocessor.tokenize(tweet)
        # 对每个分词进行标准化处理
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        # 替换特定的缩写形式
        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        # 替换时间表达式中的点
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        # 返回标准化后的 Tweet
        return " ".join(normTweet.split())
    # 标准化推特中的标记
    def normalizeToken(self, token):
        # 将标记转换为小写
        lowercased_token = token.lower()
        # 如果标记以'@'开头，则替换为'@USER'
        if token.startswith("@"):
            return "@USER"
        # 如果标记以'http'或'www'开头，则替换为'HTTPURL'
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        # 如果标记长度为1
        elif len(token) == 1:
            # 如果标记是特殊字符之一，则根据预定义的特殊字符映射进行替换
            if token in self.special_puncts:
                return self.special_puncts[token]
            # 如果存在表情解析器，则对标记进行表情解析，否则返回原始标记
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        # 如果标记长度大于1，则返回原始标记
        else:
            return token

    # 将标记转换为对应的标识符（整数）
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将标识符（整数）转换为对应的标记
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # 将一系列标记转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 连接所有标记，并移除所有'@@ '，然后去除两侧的空白字符
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    # 将词汇表保存到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件和合并文件的输出路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        out_merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 如果词汇表文件与输出路径不同且存在，则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，则将序列化后的词汇表模型写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        # 如果合并文件与输出路径不同，则复制合并文件到输出路径
        if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
            copyfile(self.merges_file, out_merge_file)
        # 返回保存的词汇表文件和合并文件的路径
        return out_vocab_file, out_merge_file

    # 解码标识符序列为字符串
    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)
    def add_from_file(self, f):
        """
        从文本文件加载一个预先存在的字典，并将其符号添加到该实例中。
        """
        # 如果输入参数是字符串类型
        if isinstance(f, str):
            try:
                # 尝试以 utf-8 编码打开文件
                with open(f, "r", encoding="utf-8") as fd:
                    # 递归调用 add_from_file 方法
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                # 如果文件未找到，则抛出异常
                raise fnfe
            except UnicodeError:
                # 如果检测到错误的编码，则抛出异常
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            return

        # 读取文件的所有行
        lines = f.readlines()
        # 遍历每一行
        for lineTmp in lines:
            # 去除行首尾的空格和换行符
            line = lineTmp.strip()
            # 查找行中最后一个空格的位置
            idx = line.rfind(" ")
            # 如果未找到空格，则抛出值错误异常
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            # 提取单词和计数
            word = line[:idx]
            # 将单词添加到编码器中，值为编码器当前长度
            self.encoder[word] = len(self.encoder)
# Natural Language Toolkit: Twitter Tokenizer
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Christopher Potts <cgpotts@stanford.edu>
#         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
#         Pierpaolo Pantone <> (modifications)
# URL: http://nltk.org/
# For license information, see LICENSE.TXT
#

"""
Twitter-aware tokenizer, designed to be flexible and easy to adapt to new domains and tasks. The basic logic is this:

1. The tuple regex_strings defines a list of regular expression strings.

2. The regex_strings strings are put, in order, into a compiled regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the user-supplied string, inside the tokenize() method of
   the class Tokenizer.

4. When instantiating Tokenizer objects, there is a single option: preserve_case. By default, it is set to True. If it
   is set to False, then the tokenizer will lowercase everything except for emoticons.

"""

######################################################################
#
# import regex  # https://github.com/nltk/nltk/issues/2409
# import html
#
######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most importantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# ToDo: Update with http://en.wikipedia.org/wiki/List_of_emoticons ?

# This particular element is used in a couple ways, so we define it
# with a name:
# docstyle-ignore
EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715
# docstyle-ignore
URLS = r"""            # Capture 1: entire matched URL
  (?:
  https?:                # URL protocol and colon
    (?:
      /{1,3}                # 1-3 slashes
      |                    #   or
      [a-z0-9%]                # Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |                    #   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:                    # One or more:
    [^\s()<>{}\[\]]+            # Run of non-space, non-()<>{}[]
    |                    #   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # 匹配平衡的括号，一层深度：(...(...)...)
    |
    \([^\s]+?\)                # 匹配平衡的括号，非递归：(...)
  )+
  (?:                    # 结尾为:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # 匹配平衡的括号，一层深度：(...(...)...)
    |
    \([^\s]+?\)                # 匹配平衡的括号，非递归：(...)
    |                    #   或者
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]    # 不是空格或这些标点符号
  )
  |                    # 或者，以下用于匹配裸域名:
  (?:
    (?<!@)                    # 不以 @ 开头，避免匹配 foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)                    # 不以 @ 结尾，
                            # 避免在 "foo.na@example.com" 中匹配 "foo.na"
  )
# 正则表达式列表，包含用于标记化的各种模式，如 URL、电话号码、表情符号等
REGEXPS = (
    URLS,  # 匹配 URL
    # 电话号码的正则表达式模式
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # exchange
      [ *\-.\)]*
      \d{4}          # base
    )""",
    # ASCII 表情符号的正则表达式模式
    EMOTICONS,
    # HTML 标签的正则表达式模式
    r"""<[^>\s]+>""",
    # ASCII 箭头符号的正则表达式模式
    r"""[\-]+>|<[\-]+""",
    # Twitter 用户名的正则表达式模式
    r"""(?:@[\w_]+)""",
    # Twitter 主题标签的正则表达式模式
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # 电子邮件地址的正则表达式模式
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",
    # docstyle-ignore
    # 剩余的词类型的正则表达式模式，如带撇号或短横线的单词、数字（包括分数、小数）、普通单词、省略号、其他非空白字符等
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # 带撇号或短横线的单词
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # 数字，包括分数、小数
    |
    (?:[\w_]+)                     # 普通单词
    |
    (?:\.(?:\s*\.){1,})            # 省略号
    |
    (?:\S)                         # 其他非空白字符
    """,
)

######################################################################
# This is the core tokenizing regex:
# 这是核心的标记化正则表达式：

# 核心的标记化正则表达式，用于将文本分割成单词
WORD_RE = regex.compile(r"""(%s)""" % "|".join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)

# WORD_RE 在以下模式上表现不佳：
# 用于检测文本中连续出现三次以上相同字符的正则表达式
HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

# 表情符号字符串单独用一个正则表达式，以便根据需要保留其大小写：
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)

# 用于将 HTML 实体规范化为 Unicode 的函数：
ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")


######################################################################
# Functions for converting html entities
######################################################################

# 将字符串转换为 Unicode
def _str_to_unicode(text, encoding=None, errors="strict"):
    if encoding is None:
        encoding = "utf-8"
    if isinstance(text, bytes):
        return text.decode(encoding, errors)
    return text


# 替换 HTML 实体
def _replace_html_entities(text, keep=(), remove_illegal=True, encoding="utf-8"):
    """
    Remove entities from text by converting them to their corresponding unicode character.

    Args:
        text:
            A unicode string or a byte string encoded in the given *encoding* (which defaults to 'utf-8').
        keep (list):
            List of entity names which should not be replaced. This supports both numeric entities (`&#nnnn;` and
            `&#hhhh;`) and named entities (such as `&nbsp;` or `&gt;`).
        remove_illegal (bool):
            If `True`, entities that can't be converted are removed. Otherwise, entities that can't be converted are
            kept "as is".

    Returns: A unicode string with the entities removed.

    See https://github.com/scrapy/w3lib/blob/master/w3lib/html.py

    Examples:

    ```py

    ```
    """
    # 导入_replace_html_entities函数
    >>> from nltk.tokenize.casual import _replace_html_entities

    # 使用_replace_html_entities函数替换HTML实体，返回替换后的字符串
    >>> _replace_html_entities(b"Price: &pound;100")
    'Price: \\xa3100'

    # 打印使用_replace_html_entities函数替换HTML实体后的字符串
    >>> print(_replace_html_entities(b"Price: &pound;100"))
    Price: £100
    ```py

    # 定义_convert_entity函数，用于替换HTML实体
    def _convert_entity(match):
        # 获取HTML实体的内容
        entity_body = match.group(3)
        # 判断是否为数字实体
        if match.group(1):
            try:
                # 尝试将实体内容转换为数字
                if match.group(2):
                    number = int(entity_body, 16)
                else:
                    number = int(entity_body, 10)
                # 处理特殊范围的数字实体
                if 0x80 <= number <= 0x9F:
                    return bytes((number,)).decode("cp1252")
            except ValueError:
                number = None
        else:
            # 处理非数字实体
            if entity_body in keep:
                return match.group(0)
            else:
                number = html.entities.name2codepoint.get(entity_body)
        # 如果存在数字实体，尝试转换为字符
        if number is not None:
            try:
                return chr(number)
            except (ValueError, OverflowError):
                pass

        # 如果无法转换或需要移除非法字符，则返回空字符串或原实体
        return "" if remove_illegal else match.group(0)

    # 使用ENT_RE正则表达式替换HTML实体，返回替换后的Unicode字符串
    return ENT_RE.sub(_convert_entity, _str_to_unicode(text, encoding))
class TweetTokenizer:
    r"""
    Examples:

    ```python
    >>> # Tokenizer for tweets.
    >>> from nltk.tokenize import TweetTokenizer

    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

    >>> # Examples using *strip_handles* and *reduce_len parameters*:
    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = "@remy: This is waaaaayyyy too much for you!!!!!!"
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
    ```py"""

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        # 是否保持单词大小写
        self.preserve_case = preserve_case
        # 是否缩短重复字符序列
        self.reduce_len = reduce_len
        # 是否去除 Twitter 用户名
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        Args:
            text: str

        Returns: list(str) A tokenized list of strings; concatenating this list returns the original string if
        `preserve_case=False`
        """
        # 修复 HTML 字符实体
        text = _replace_html_entities(text)
        # 去除用户名句柄
        if self.strip_handles:
            text = remove_handles(text)
        # 标准化单词长度
        if self.reduce_len:
            text = reduce_lengthening(text)
        # 缩短有问题的字符序列
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # 分词
        words = WORD_RE.findall(safe_text)
        # 可能改变大小写，但避免改变表情符号如 :D 变成 :d:
        if not self.preserve_case:
            words = [x if EMOTICON_RE.search(x) else x.lower() for x in words]
        return words


######################################################################
# Normalization Functions
######################################################################


def reduce_lengthening(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences of length 3.
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


def remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    # 用空格替换句柄，以确保句柄周围的文本被正确分词
    return pattern.sub(" ", text)


######################################################################
# Tokenization Function
######################################################################


def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    """
    # 用于包装分词器的便利函数
    """
    # 返回一个 TweetTokenizer 对象，并对文本进行分词处理
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles).tokenize(
        text
    )
###############################################################################

# 导入numpy库，命名为np
import numpy as np
# 导入pandas库，命名为pd
import pandas as pd
# 导入matplotlib库的pyplot模块，命名为plt
import matplotlib.pyplot as plt

# 定义一个函数calc_sqrt，接受一个参数x
def calc_sqrt(x):
    # 使用numpy的sqrt函数计算x的平方根，并返回结果
    return np.sqrt(x)

# 创建一个Series对象，其中包含一组数字
data = pd.Series([1, 2, 3, 4, 5])

# 调用Series对象的apply方法，应用calc_sqrt函数到每个元素上，并得到一个新的Series对象
result = data.apply(calc_sqrt)

# 使用matplotlib绘制图形，绘制原始数据和计算结果
plt.plot(data, label='Original Data')
plt.plot(result, label='Square Root')
plt.legend()
plt.show()
```