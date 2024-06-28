# `.\models\bertweet\tokenization_bertweet.py`

```
# 导入标准库和第三方库
import html  # 用于 HTML 编码和解码
import os    # 提供与操作系统交互的功能
import re    # 用于正则表达式操作
from shutil import copyfile  # 用于复制文件
from typing import List, Optional, Tuple  # 引入类型提示相关的库

import regex  # 引入 regex 库，支持更强大的正则表达式功能

# 导入 Tokenizer 的基类 PreTrainedTokenizer 和日志模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件和合并文件的名称映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "merges_file": "bpe.codes",
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/bertweet-base": "https://huggingface.co/vinai/bertweet-base/resolve/main/vocab.txt",
    },
    "merges_file": {
        "vinai/bertweet-base": "https://huggingface.co/vinai/bertweet-base/resolve/main/bpe.codes",
    },
}

# 预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vinai/bertweet-base": 128,
}

def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词被表示为符号元组（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class BertweetTokenizer(PreTrainedTokenizer):
    """
    构造一个 BERTweet 分词器，使用字节对编码。

    此分词器继承自 PreTrainedTokenizer，该类包含大多数主要方法。用户应参考这个超类以获取更多关于这些方法的信息。
    """
    # 定义一个 Transformer 模型的配置类，用于管理与模型相关的参数和配置
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 初始化函数，用于设置模型配置参数
    def __init__(
        self,
        vocab_file,  # 词汇表文件的路径
        merges_file,  # 合并文件的路径
        normalization=False,  # 是否进行标准化预处理，默认为False
        bos_token="<s>",  # 预训练期间用于序列开始的特殊符号，默认为"<s>"
        eos_token="</s>",  # 序列结束的特殊符号，默认为"</s>"
        sep_token="</s>",  # 用于多个序列构建时的分隔符，默认为"</s>"
        cls_token="<s>",  # 序列分类时使用的特殊符号，构建时是序列的第一个符号，默认为"<s>"
        unk_token="<unk>",  # 未知符号，词汇表中没有时的替代符号，默认为"<unk>"
        pad_token="<pad>",  # 填充符号，用于处理不同长度序列时的填充，默认为"<pad>"
        mask_token="<mask>",  # 掩码符号，用于掩码语言建模训练中的标记，默认为"<mask>"
        **kwargs,  # 其他可选参数
    ):
        try:
            from emoji import demojize  # 尝试导入 demojize 函数从 emoji 模块
            self.demojizer = demojize  # 如果成功导入，将 demojize 函数赋值给 self.demojizer
        except ImportError:
            logger.warning(
                "emoji is not installed, thus not converting emoticons or emojis into text. Install emoji: pip3"
                " install emoji==0.6.0"
            )
            self.demojizer = None  # 如果导入失败，记录警告信息，并将 self.demojizer 设为 None

        self.vocab_file = vocab_file  # 初始化词汇表文件路径
        self.merges_file = merges_file  # 初始化合并文件路径

        self.encoder = {}  # 初始化编码器字典
        self.encoder[str(bos_token)] = 0  # 将特殊标记 bos_token 编码为 0
        self.encoder[str(pad_token)] = 1  # 将特殊标记 pad_token 编码为 1
        self.encoder[str(eos_token)] = 2  # 将特殊标记 eos_token 编码为 2
        self.encoder[str(unk_token)] = 3  # 将特殊标记 unk_token 编码为 3

        self.add_from_file(vocab_file)  # 调用 add_from_file 方法，从 vocab_file 添加更多词汇到编码器

        self.decoder = {v: k for k, v in self.encoder.items()}  # 创建解码器，将编码器的键值对颠倒

        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]  # 读取并处理合并文件的内容
        merges = [tuple(merge.split()[:-1]) for merge in merges]  # 将每行合并内容转换为元组列表
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # 创建 BPE 合并的排名字典
        self.cache = {}  # 初始化缓存字典

        self.normalization = normalization  # 设置文本规范化选项
        self.tweetPreprocessor = TweetTokenizer()  # 初始化 TweetTokenizer 作为 tweetPreprocessor
        self.special_puncts = {"’": "'", "…": "..."}  # 定义特殊标点符号映射

        super().__init__(  # 调用父类的初始化方法，传递相应参数和关键字参数
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
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
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

        # If the token list already has special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there are no sequence pairs (token_ids_1 is None), add special tokens around token_ids_0
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # For sequence pairs, add special tokens around both token_ids_0 and token_ids_1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BERTweet does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """

        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there are no sequence pairs, return a list of zeros of length equal to cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # For sequence pairs, return a list of zeros of length equal to cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Return the size of the vocabulary, which is the length of the encoder dictionary
        return len(self.encoder)

    def get_vocab(self):
        # Return the combined dictionary of encoder and added_tokens_encoder
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 如果 token 已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        # 将 token 转换为元组形式
        word = tuple(token)
        # 在 token 的末尾添加 "</w>"，表示单词结束
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        # 获取单词中的所有字符对，并进行 BPE 算法处理
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回原始 token
        if not pairs:
            return token

        # 循环处理字符对，直到无法再合并为止
        while True:
            # 找到优先级最高的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不在预定义的 BPE 优先级中，停止处理
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历单词中的字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到字符对的第一个字符，直接将剩余部分添加到新单词中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将当前位置到字符对第一个字符位置之间的部分添加到新单词中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前位置的字符与字符对的第一个字符相同，并且下一个字符与字符对的第二个字符相同，则合并为一个新的字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则，将当前位置的字符添加到新单词中，并移动到下一个位置
                    new_word.append(word[i])
                    i += 1
            # 将新单词转换为元组形式，并更新 word 变量为新单词
            new_word = tuple(new_word)
            word = new_word
            # 如果新单词长度为1，停止循环
            if len(word) == 1:
                break
            else:
                # 否则，继续获取新的字符对
                pairs = get_pairs(word)
        
        # 将处理后的单词以 "@@ " 连接起来，并去掉末尾的特殊标记 "</w>"
        word = "@@ ".join(word)
        word = word[:-4]
        # 将处理后的结果缓存起来，并返回
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 如果启用了 Tweet 规范化，则在进行 BPE 处理之前先对文本进行规范化
        if self.normalization:
            text = self.normalizeTweet(text)

        split_tokens = []
        # 使用正则表达式将文本分割成单词列表
        words = re.findall(r"\S+\n?", text)
        for token in words:
            # 对每个单词进行 BPE 处理，并将处理结果按空格分割后添加到 split_tokens 列表中
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    def normalizeTweet(self, tweet):
        """
        Normalize a raw Tweet
        """
        # 替换 Tweet 中的特殊标点符号
        for punct in self.special_puncts:
            tweet = tweet.replace(punct, self.special_puncts[punct])

        # 使用 Tweet 预处理器对 Tweet 进行分词
        tokens = self.tweetPreprocessor.tokenize(tweet)
        # 对每个 token 进行规范化处理，并用空格连接起来
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        # 进行特定的单词规范化处理，替换常见的缩写和缩略语
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
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())
    # 将给定的 token 标准化为小写形式
    def normalizeToken(self, token):
        lowercased_token = token.lower()
        # 如果 token 以 "@" 开头，则返回 "@USER"
        if token.startswith("@"):
            return "@USER"
        # 如果 token 的小写形式以 "http" 或 "www" 开头，则返回 "HTTPURL"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        # 如果 token 的长度为 1
        elif len(token) == 1:
            # 如果 token 是特殊标点符号中的一种，则返回其对应的值
            if token in self.special_puncts:
                return self.special_puncts[token]
            # 如果存在表情解析器，则用表情解析器处理 token，否则返回原 token
            if self.demojizer is not None:
                return self.demojizer(token)
            else:
                return token
        # 对于其他情况，直接返回 token
        else:
            return token

    # 根据 token 转换为对应的 id，使用给定的词汇表
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 根据 id 转换为对应的 token，使用给定的词汇表
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    # 将一系列 tokens 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构造词汇表文件路径和合并文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        out_merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        
        # 如果当前词汇表文件路径与目标路径不同且当前路径下存在词汇表文件，则复制词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前路径下不存在词汇表文件，则将当前模型的序列化词汇表模型写入目标路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 如果当前合并文件路径与目标路径不同，则复制合并文件到目标路径
        if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
            copyfile(self.merges_file, out_merge_file)

        return out_vocab_file, out_merge_file
    def add_from_file(self, f):
        """
        从文本文件中加载一个预先存在的字典，并将其符号添加到当前实例中。
        """
        # 如果输入参数 f 是字符串类型，则尝试打开该文件
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    # 递归调用 add_from_file 方法，加载文件内容
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                # 如果文件不存在，则抛出 FileNotFound 异常
                raise fnfe
            except UnicodeError:
                # 如果在文件中检测到不正确的编码，则抛出异常
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            # 返回，结束当前函数调用
            return

        # 读取文件中的所有行
        lines = f.readlines()
        # 遍历每一行内容
        for lineTmp in lines:
            # 去除行首尾空白符
            line = lineTmp.strip()
            # 查找行中最后一个空格的位置
            idx = line.rfind(" ")
            # 如果找不到空格，则抛出数值错误异常
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            # 提取空格之前的部分作为单词
            word = line[:idx]
            # 将单词作为键，将当前编码器长度作为值存入编码器字典中
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
      [<>]?                           # optional opening angle bracket
      [:;=8]                          # eyes
      [\-o\*\']?                      # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\]      # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\]      # mouth
      [\-o\*\']?                      # optional nose
      [:;=8]                          # eyes
      [<>]?                           # optional closing angle bracket
      |
      <3                               # heart
    )"""

# URL pattern due to John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715
# docstyle-ignore
URLS = r"""            # Capture 1: entire matched URL
  (?:
  https?:                     # URL protocol and colon
    (?:
      /{1,3}                     # 1-3 slashes
      |                         #   or
      [a-z0-9%]                     # Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |                         #   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:                         # One or more:
    [^\s()<>{}\[\]]+                 # Run of non-space, non-()<>{}[]
    |                         #   or

    \(
      [^\s()<>{}\[\]]+
    \)
  )+
  (?:                         # End with:
    \(
      [^\s()<>{}\[\]]+
    \)
    |                         #   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]
  )
"""

# The above pattern defines URLs using a regex for tokenization purposes,
# covering various formats and components typically found in URLs.
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # 匹配具有平衡括号的一级深度的表达式：(...(...)...)
    |
    \([^\s]+?\)                # 匹配非递归的平衡括号表达式：(...)
  )+                          # 上述两种模式可以出现一次或多次，即匹配多个括号嵌套或单个括号
  (?:                          # 结尾处可以是以下模式之一：
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # 匹配具有平衡括号的一级深度的表达式：(...(...)...)
    |
    \([^\s]+?\)                # 匹配非递归的平衡括号表达式：(...)
    |                          # 或者
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]    # 不是空格或特定的标点字符
  )
  |                          # 或者，用于匹配裸域名：
  (?:
    (?<!@)                    # 前面不是 @，避免在电子邮件地址中匹配例如 "foo@_gmail.com_"
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)                    # 后面不是 @，避免在电子邮件地址中匹配例如 "foo.na" 在 "foo.na@example.com" 中
  )


这段代码是一个正则表达式模式，用于匹配具有特定形式的括号结构和裸域名。
# 定义正则表达式模式以识别不同类型的标记
# 包括 URL、电话号码、ASCII 表情、HTML 标签、ASCII 箭头、Twitter 用户名、Twitter 主题标签、电子邮件地址等
REGEXPS = (
    URLS,  # 匹配 URL
    r"""
    (?:
      (?:            # (国际)
        \+?[01]
        [ *\-.\)]*
      )?
      (?:            # (区号)
        [\(]?
        \d{3}
        [ *\-.\)]*
      )?
      \d{3}          # 交换机
      [ *\-.\)]*
      \d{4}          # 基站
    )""",  # 匹配电话号码
    EMOTICONS,  # 匹配 ASCII 表情
    r"""<[^>\s]+>""",  # 匹配 HTML 标签
    r"""[\-]+>|<[\-]+""",  # 匹配 ASCII 箭头
    r"""(?:@[\w_]+)""",  # 匹配 Twitter 用户名
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",  # 匹配 Twitter 主题标签
    r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]""",  # 匹配电子邮件地址
    r"""
    (?:[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]) # 带有撇号或破折号的单词
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # 数字，包括分数、小数点
    |
    (?:[\w_]+)                     # 没有撇号或破折号的单词
    |
    (?:\.(?:\s*\.){1,})            # 省略号
    |
    (?:\S)                         # 其他非空白字符
    """,  # 匹配剩余的词类
)

######################################################################
# 这是核心的分词正则表达式:

# 将 REGEXPS 中的所有模式组合成一个大的正则表达式
WORD_RE = regex.compile(r"""(%s)""" % "|".join(REGEXPS), regex.VERBOSE | regex.I | regex.UNICODE)

# HANG_RE 用于识别连续字符的模式
HANG_RE = regex.compile(r"([^a-zA-Z0-9])\1{3,}")

# EMOTICON_RE 用于识别表情符号的模式
EMOTICON_RE = regex.compile(EMOTICONS, regex.VERBOSE | regex.I | regex.UNICODE)

# ENT_RE 用于将 HTML 实体转换为 Unicode 字符的模式
ENT_RE = regex.compile(r"&(#?(x?))([^&;\s]+);")
    # 导入HTML实体替换函数
    from nltk.tokenize.casual import _replace_html_entities

    # 使用HTML实体替换函数处理包含HTML实体的字节字符串，返回替换后的字符串
    _replace_html_entities(b"Price: &pound;100")
    # 输出结果：'Price: \\xa3100'

    # 打印使用HTML实体替换函数处理包含HTML实体的字节字符串，应该输出替换后的Unicode字符串
    print(_replace_html_entities(b"Price: &pound;100"))
    # 输出结果：Price: £100
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
    ```"""

    def __init__(self, preserve_case=True, reduce_len=False, strip_handles=False):
        # Initialize the TweetTokenizer with options to preserve case, reduce elongated words, and strip handles.
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, text):
        """
        Tokenize a given text into a list of words.

        Args:
            text: str

        Returns:
            list(str): A list of tokens extracted from the text.
        """
        # Fix HTML character entities before tokenization
        text = _replace_html_entities(text)
        # Remove Twitter handles if strip_handles is enabled
        if self.strip_handles:
            text = remove_handles(text)
        # Reduce elongated words to their base form if reduce_len is enabled
        if self.reduce_len:
            text = reduce_lengthening(text)
        # Replace problematic sequences of characters for safe tokenization
        safe_text = HANG_RE.sub(r"\1\1\1", text)
        # Tokenize the text using a regular expression for word boundaries
        words = WORD_RE.findall(safe_text)
        # Adjust word case unless it is part of an emoticon to preserve emoticon capitalization
        if not self.preserve_case:
            words = [x if EMOTICON_RE.search(x) else x.lower() for x in words]
        return words


######################################################################
# Normalization Functions
######################################################################

def reduce_lengthening(text):
    """
    Reduce repeated character sequences of length 3 or greater to sequences of length 3.

    Args:
        text: str

    Returns:
        str: Text with reduced elongations.
    """
    pattern = regex.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1\1", text)


def remove_handles(text):
    """
    Remove Twitter username handles from text.

    Args:
        text: str

    Returns:
        str: Text with removed handles replaced by spaces.
    """
    pattern = regex.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)"
    )
    # Substitute handles with ' ' to ensure correct tokenization around removed handles
    return pattern.sub(" ", text)


######################################################################
# Tokenization Function
######################################################################

def casual_tokenize(text, preserve_case=True, reduce_len=False, strip_handles=False):
    """
    Tokenize a text string using casual tokenization rules.

    Args:
        text: str
        preserve_case: bool, optional (default=True)
            Whether to preserve case in tokens.
        reduce_len: bool, optional (default=False)
            Whether to reduce elongated words.
        strip_handles: bool, optional (default=False)
            Whether to remove Twitter handles.

    Returns:
        list(str): A list of tokens extracted from the text based on specified rules.
    """
    # 创建一个TweetTokenizer对象，用于分词化处理，根据参数设置保留大小写、缩短长度和去除句柄
    """
    Convenience function for wrapping the tokenizer.
    """
    # 返回通过TweetTokenizer对象对文本进行分词化处理得到的结果
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len, strip_handles=strip_handles).tokenize(
        text
    )
###############################################################################

# 定义一个函数 `calculate_total`，接收一个参数 `items`
def calculate_total(items):
    # 初始化一个变量 `total`，用于累计总和
    total = 0
    # 遍历参数 `items` 中的每个元素，将其加到 `total` 中
    for item in items:
        total += item
    # 返回累计的总和 `total`
    return total
```