# `.\models\fsmt\tokenization_fsmt.py`

```
# 对代码文件进行编码声明为UTF-8
# 版权声明
# 根据Apache许可版本2.0授权，除非符合许可条件，否则不得使用此文件
# 你可以获取许可的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可分发的软件是基于"AS IS"基础分发的，没有任何担保或条件，无论明示或暗示
# 请参见许可协议获取特定语言的授权和限制
# FSMT的标记化类

# 导入所需的库

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {
    "src_vocab_file": "vocab-src.json",
    "tgt_vocab_file": "vocab-tgt.json",
    "merges_file": "merges.txt",
}

# 预训练的词汇文件
PRETRAINED_VOCAB_FILES_MAP = {
    "src_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-src.json"
    },
    "tgt_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-tgt.json"
    },
    "merges_file": {"stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/merges.txt"},
}

# 预训练的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"stas/tiny-wmt19-en-de": 1024}
PRETRAINED_INIT_CONFIGURATION = {
    "stas/tiny-wmt19-en-de": {
        "langs": ["en", "de"],
        "model_max_length": 1024,
        "special_tokens_map_file": None,
        "full_tokenizer_file": None,
    }
}

# 获取词典中的词对
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 替换Unicode标点符号
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    # 将全角的数字替换为半角的数字
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    
    # 将文本中的中文句号后的空格替换为标准的英文句号后的空格
    text = re.sub(r"．\s*", ". ", text)
    
    # 替换其他一些特殊字符
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    
    # 返回处理后的文本
    return text
# 定义一个函数，用于去除给定文本中不可打印的字符
def remove_non_printing_char(text):
    # 这是一个移植自 GitHub 上某个 Perl 脚本的函数
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    # 创建一个空列表，用于存储过滤后的字符
    output = []
    # 遍历给定文本中的每个字符
    for char in text:
        # 获取该字符的 Unicode 类别
        cat = unicodedata.category(char)
        # 如果类别以 "C" 开头，意味着这是一个不可打印的字符
        if cat.startswith("C"):
            # 跳过不可打印的字符
            continue
        # 将可打印的字符添加到输出列表中
        output.append(char)
    # 将列表中的字符连接成一个字符串并返回
    return "".join(output)


# 下面是关于 FSMTTokenizer 类的移植说明
# FSMTTokenizer 类基于 XLMTokenizer 建模
#
# 添加了以下属性：
# - 源语言词汇表文件
# - 目标语言词汇表文件
# - 语言对列表


# 定义 FSMTTokenizer 类，继承自预训练的分词器基类
class FSMTTokenizer(PreTrainedTokenizer):
    # 这个分词器用于构建 FAIRSEQ Transformer 分词器，基于字节对编码(BPE)
    """
    Construct an FAIRSEQ Transformer tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses 预处理和分词。
    - 对所有输入文本进行标准化处理。
    - `special_tokens` 参数和 `set_special_tokens` 函数可以用于向词汇表中添加额外符号，例如 "__classify__"。
    - `langs` 参数定义了一对语言。
    
    这个分词器继承自 [`PreTrainedTokenizer`]，其中包含了大多数主要方法。用户应参考这个基类以了解更多关于这些方法的信息。

    Args:
        langs (`List[str]`, *optional*):
            两个语言的列表，分别用于翻译的源语言和目标语言，例如 `["en", "ru"]`。
        src_vocab_file (`str`, *optional*):
            源语言词汇表文件的文件名。
        tgt_vocab_file (`str`, *optional*):
            目标语言词汇表文件的文件名。
        merges_file (`str`, *optional*):
            合并文件的文件名。
        do_lower_case (`bool`, *optional*, defaults to `False`):
            是否在分词时将输入文本转为小写。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知词汇标记。当词汇不在词汇表中时，不能转换为 ID，就会用这个标记替代。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            序列开始标记，用于预训练过程中。可以用作序列分类器的标记。
            
            <Tip>

            构建序列时，特别标记通常不是序列开始时使用的标记。构建序列开始时用的是 `cls_token`。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔符号，用于将多个序列组合成一个序列，例如用于序列分类或用于文本和问题的问答。它也用于一个由特殊标记构建的序列的末尾。
        pad_token (`str`, *optional*, defaults到 `"<pad>"`):
            填充标记，例如用于不同长度的序列进行批处理。
    """

    # 指定词汇表文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 指定预训练词汇表文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练初始化配置赋值给变量pretrained_init_configuration
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 将最大模型输入大小赋值给变量max_model_input_sizes
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法，接受多个参数
    def __init__(
        self,
        langs=None,
        src_vocab_file=None,
        tgt_vocab_file=None,
        merges_file=None,
        do_lower_case=False,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        # 尝试导入sacremoses库，若导入失败则抛出ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将sacremoses库赋值给self.sm
        self.sm = sacremoses

        # 初始化参数赋值
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file
        self.merges_file = merges_file
        self.do_lower_case = do_lower_case

        # 缓存sm.MosesPunctNormalizer实例
        self.cache_moses_punct_normalizer = {}
        # 缓存sm.MosesTokenizer实例
        self.cache_moses_tokenizer = {}
        self.cache_moses_detokenizer = {}

        # 若langs参数存在且长度为2，则赋值给self.src_lang和self.tgt_lang，否则抛出ValueError异常
        if langs and len(langs) == 2:
            self.src_lang, self.tgt_lang = langs
        else:
            raise ValueError(
                f"arg `langs` needs to be a list of 2 langs, e.g. ['en', 'ru'], but got {langs}. "
                "Usually that means that tokenizer can't find a mapping for the given model path "
                "in PRETRAINED_VOCAB_FILES_MAP, and other maps of this tokenizer."
            )

        # 读取源语言词汇文件，加载到self.encoder
        with open(src_vocab_file, encoding="utf-8") as src_vocab_handle:
            self.encoder = json.load(src_vocab_handle)
        # 读取目标语言词汇文件，加载到tgt_vocab，然后生成self.decoder
        with open(tgt_vocab_file, encoding="utf-8") as tgt_vocab_handle:
            tgt_vocab = json.load(tgt_vocab_handle)
            self.decoder = {v: k for k, v in tgt_vocab.items()}
        # 读取merge文件，生成self.bpe_ranks
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        # 调用父类初始化方法
        super().__init__(
            langs=langs,
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            **kwargs,
        )

    # 覆盖方式获取词汇表方法，返回值为源语言词汇表
    def get_vocab(self) -> Dict[str, int]:
        return self.get_src_vocab()

    # 覆盖方式获取词汇表大小，返回值为源语言词汇表大小
    @property
    def vocab_size(self) -> int:
        return self.src_vocab_size
    # 使用 Moses 格式对文本进行标点符号规范化处理
    def moses_punct_norm(self, text, lang):
        # 如果语言不在缓存中，则创建 Moses 格式标点符号规范化器并加入缓存
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        # 返回规范化后的文本
        return self.cache_moses_punct_normalizer[lang].normalize(text)

    # 使用 Moses 格式对文本进行分词处理
    def moses_tokenize(self, text, lang):
        # 如果语言不在缓存中，则创建 Moses 格式分词器并加入缓存
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        # 返回分词结果
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    # 使用 Moses 格式对分词结果进行反标点符号化处理
    def moses_detokenize(self, tokens, lang):
        # 如果语言不在缓存中，则创建 Moses 格式反标点符号化器并加入缓存
        if lang not in self.cache_moses_detokenizer:
            moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        # 返回反标点符号化后的文本
        return self.cache_moses_detokenizer[lang].detokenize(tokens)

    # 使用 Moses 格式处理文本的整个流程：标点符号规范化、分词、去除非打印字符
    def moses_pipeline(self, text, lang):
        # 替换文本中的 Unicode 标点符号
        text = replace_unicode_punct(text)
        # 进行 Moses 格式的标点符号规范化处理
        text = self.moses_punct_norm(text, lang)
        # 去除文本中的非打印字符
        text = remove_non_printing_char(text)
        # 返回处理后的文本
        return text

    # 返回源语言词汇表的大小
    @property
    def src_vocab_size(self):
        return len(self.encoder)

    # 返回目标语言词汇表的大小
    @property
    def tgt_vocab_size(self):
        return len(self.decoder)

    # 获取源语言的词汇表，包括新增的标记
    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 获取目标语言的词汇表，包括新增的标记
    def get_tgt_vocab(self):
        return dict(self.decoder, **self.added_tokens_decoder)

    # 使用 BPE 算法对单词进行分词
    def bpe(self, token):
        # 将单词转换为 BPE 格式
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果单词已经在缓存中，则直接返回结果
        if token in self.cache:
            return self.cache[token]
        # 获取单词的所有可能的分词对
        pairs = get_pairs(word)

        # 如果没有分词对，则将单词添加结束符号后返回
        if not pairs:
            return token + "</w>"

        while True:
            # 选择当前权重最小的分词对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果分词对不在权重表中，则停止循环
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
        word = " ".join(word)
        # 处理特殊情况下的单词格式
        if word == "\n  </w>":
            word = "\n</w>"
        # 将处理结果加入缓存并返回
        self.cache[token] = word
        return word
    def _tokenize(self, text, lang="en", bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported
              languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        # 如果用户传入的语言参数不是源语言，则将语言参数设为源语言
        # ignore `lang` which is currently isn't explicitly passed in tokenization_utils.py and always results in lang=en
        # if lang != self.src_lang:
        #     raise ValueError(f"Expected lang={self.src_lang}, but got {lang}")
        lang = self.src_lang

        # 如果设置了小写化标志，则将文本转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 如果设置了绕过分词器的标志，则直接按空格分割文本
        if bypass_tokenizer:
            text = text.split()
        else:
            # 否则使用 Moses 分词器进行分词
            text = self.moses_pipeline(text, lang=lang)
            # 使用 Moses 分词器进行标记化
            text = self.moses_tokenize(text, lang=lang)

        # 对标记化后的结果进行 BPE 处理
        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将给定的 token 转换为对应的 id，如果不存在则返回未知 token 的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将给定的 id 转换为对应的 token，如果不存在则返回未知 token
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""

        # 移除 BPE 标记并重新连接为单个字符串
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # 使用 Moses 的反标记化函数将 token 序列转换为文本
        text = self.moses_detokenize(tokens, self.tgt_lang)
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        根据输入的序列或序列对构建模型输入，用于序列分类任务，通过拼接和添加特殊标记。FAIRSEQ Transformer序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表。

        Returns:
            `List[int]`: 包含适当特殊标记的输入ID列表。
        """
        sep = [self.sep_token_id]

        # fairseq 中不使用 bos
        if token_ids_1 is None:
            return token_ids_0 + sep
        return token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的 token 列表中检索序列ID。当使用 tokenizer 的`prepare_for_model`方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                token 列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个在范围[0, 1]内的整数列表：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        # fairseq 中不使用 bos
        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An
        FAIRSEQ_TRANSFORMER sequence pair mask has the following format:
        """
        sep = [self.sep_token_id]

        # no bos used in fairseq
        # 如果token_ids_1为None，则返回仅包含第一部分mask的列表（全为0）
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查save_directory是否为目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 定义保存路径和文件名前缀
        src_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["src_vocab_file"]
        )
        tgt_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["tgt_vocab_file"]
        )
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器数据写入源词汇文件
        with open(src_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 将解码器数据写入目标词汇文件
        with open(tgt_vocab_file, "w", encoding="utf-8") as f:
            tgt_vocab = {v: k for k, v in self.decoder.items()}
            f.write(json.dumps(tgt_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 将BPE标记数据写入合并文件
        index = 0
        with open(merges_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merges_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的源词汇文件、目标词汇文件和合并文件路径
        return src_vocab_file, tgt_vocab_file, merges_file
    # 定义一个特殊方法，用于获取对象的状态
    def __getstate__(self):
        # 复制对象的字典状态
        state = self.__dict__.copy()
        # 将对象的某个属性设置为 None
        state["sm"] = None
        # 返回状态字典
        return state

    # 定义一个特殊方法，用于设置对象的状态
    def __setstate__(self, d):
        # 将对象的字典状态设置为给定的状态
        self.__dict__ = d

        # 尝试导入 sacremoses 模块
        try:
            import sacremoses
        # 如果导入失败，则抛出 ImportError 异常
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )
        # 将导入的 sacremoses 模块赋值给对象的属性
        self.sm = sacremoses
```