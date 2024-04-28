# `.\models\flaubert\tokenization_flaubert.py`

```py
# 导入所需模块
import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

# 导入父类中的 PreTrainedTokenizer 类
from ...tokenization_utils import PreTrainedTokenizer
# 导入日志模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "flaubert/flaubert_small_cased": (
            "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/vocab.json"
        ),
        "flaubert/flaubert_base_uncased": (
            "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/vocab.json"
        ),
        "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/vocab.json",
        "flaubert/flaubert_large_cased": (
            "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "flaubert/flaubert_small_cased": (
            "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/merges.txt"
        ),
        "flaubert/flaubert_base_uncased": (
            "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/merges.txt"
        ),
        "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/merges.txt",
        "flaubert/flaubert_large_cased": (
            "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/merges.txt"
        ),
    },
}

# 定义预训练位置嵌入的尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "flaubert/flaubert_small_cased": 512,
    "flaubert/flaubert_base_uncased": 512,
    "flaubert/flaubert_base_cased": 512,
    "flaubert/flaubert_large_cased": 512,
}

# 定义预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "flaubert/flaubert_small_cased": {"do_lowercase": False},
    "flaubert/flaubert_base_uncased": {"do_lowercase": True},
    "flaubert/flaubert_base_cased": {"do_lowercase": False},
    "flaubert/flaubert_large_cased": {"do_lowercase": False},
}

# 定义一个函数，将输入文本转换为 Unicode 编码（如果尚未转换），假定输入是 UTF-8 格式的
def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # 定义函数 ensure_text，用于确保输入的字符串是文本格式
    def ensure_text(s, encoding="utf-8", errors="strict"):
        # 如果输入是字节流，则按照指定编码和错误处理方式解码成文本
        if isinstance(s, bytes):
            return s.decode(encoding, errors)
        # 如果输入已经是文本，则直接返回
        elif isinstance(s, str):
            return s
        # 如果输入既不是字节流也不是文本，则抛出类型错误
        else:
            raise TypeError(f"not expecting type '{type(s)}'")
    
    # 调用 ensure_text 函数，将输入文本以指定编码解码并忽略错误，然后返回结果
    return ensure_text(text, encoding="utf-8", errors="ignore")
# 从单词中获取符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    # 遍历单词中的每个符号对，将符号对添加到集合中
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# 替换文本中的 Unicode 标点符号
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    # 替换特定的 Unicode 标点符号为普通字符
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
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


# 移除文本中的不可打印字符
def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []
    # 遍历文本中的每个字符，移除不可打印字符
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            continue
        output.append(char)
    return "".join(output)


class FlaubertTokenizer(PreTrainedTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    """
    为 Hugface Transformers 模型构建一个 token 自定义词表类，继承自 VocabWithMerge
    继承这个父类，以便获取更多关于这些方法的信息。

    Args:
        vocab_file (`str`):
            词汇表文件。
        merges_file (`str`):
            合并文件。
        do_lowercase (`bool`, *可选*, 默认为 `False`):
            控制是否小写处理。
        unk_token (`str`, *可选*, 默认为 `"<unk>"`):
            未知令牌。词汇表中不存在的令牌不能转换为 ID，而是设置为此令牌。
        bos_token (`str`, *可选*, 默认为 `"<s>"`):
            在预训练期间使用的序列开始令牌。可以用作序列分类器令牌。

            <提示>

            在使用特殊令牌构建序列时，这不是用于序列开头的令牌。使用的令牌是 `cls_token`。

            </提示>

        sep_token (`str`, *可选*, 默认为 `"</s>"`):
            分隔符令牌，在从多个序列构建序列时使用，例如用于序列分类或用于文本和问题进行问答的两个序列。还用作使用特殊令牌构建的序列的最后一个令牌。
        pad_token (`str`, *可选*, 默认为 `"<pad>"`):
            用于填充的令牌，例如当对不同长度的序列进行批处理时。
        cls_token (`str`, *可选*, 默认为 `"</s>"`):
            用于序列分类的分类器令牌（对整个序列进行分类而不是对每个令牌进行分类）。使用特殊令牌构建时，是序列的第一个令牌。
        mask_token (`str`, *可选*, 默认为 `"<special1>"`):
            用于掩盖值的令牌。这是在使用掩盖语言建模训练此模型时使用的令牌。这是模型将尝试预测的令牌。
        additional_special_tokens (`List[str]`, *可选*, 默认为 `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            额外特殊令牌的列表。
        lang2id (`Dict[str, int]`, *可选*):
            将语言字符串标识符映射到它们的 ID 的字典。
        id2lang (`Dict[int, str]`, *可选*):
            将语言 ID 映射到它们的字符串标识符的字典。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
```  
    # 初始化方法，用于初始化 FlaubertTokenizer 对象
    def __init__(
        self,
        vocab_file,
        merges_file,
        do_lowercase=False,  # 是否将输入文本转换为小写，默认为 False
        unk_token="<unk>",  # 未知标记的字符串表示，默认为 "<unk>"
        bos_token="<s>",   # 句子的起始标记，默认为 "<s>"
        sep_token="</s>",  # 句子的分隔标记，默认为 "</s>"
        pad_token="<pad>",  # 填充标记，默认为 "<pad>"
        cls_token="</s>",  # 分类任务中的标记，默认为 "</s>"
        mask_token="<special1>",  # 掩码标记，默认为 "<special1>"
        additional_special_tokens=[  # 额外的特殊标记列表，默认为空列表
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        lang2id=None,  # 语言到 id 的映射字典，默认为 None
        id2lang=None,  # id 到语言的映射字典，默认为 None
        **kwargs,  # 其他关键字参数
    ):
        # 从关键字参数中弹出 do_lowercase_and_remove_accent，但不会产生影响
        do_lowercase_and_remove_accent = kwargs.pop("do_lowercase_and_remove_accent", None)
        # 如果 do_lowercase_and_remove_accent 不为 None，则发出警告，但不会产生影响
        if do_lowercase_and_remove_accent is not None:
            logger.warning(
                "`do_lowercase_and_remove_accent` is passed as a keyword argument, but this won't do anything."
                " `FlaubertTokenizer` will always set it to `False`."
            )
        # 将 do_lowercase_and_remove_accent 始终设置为 False
        # （即使传入了其他值，也会被忽略）
        self.do_lowercase_and_remove_accent = False

        # 设置是否将输入文本转换为小写
        self.do_lowercase = do_lowercase

        try:
            # 尝试导入 sacremoses 库
            import sacremoses
        except ImportError:
            # 如果导入失败，抛出 ImportError 异常
            raise ImportError(
                "You need to install sacremoses to use FlaubertTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 导入成功，将 sacremoses 模块赋值给 self.sm
        self.sm = sacremoses

        # 缓存 sm.MosesPunctNormalizer 实例的字典
        self.cache_moses_punct_normalizer = {}
        # 缓存 sm.MosesTokenizer 实例的字典
        self.cache_moses_tokenizer = {}
        # 自定义分词器的语言集合
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        # 初始化语言到 id 的映射字典和 id 到语言的映射字典
        self.lang2id = lang2id
        self.id2lang = id2lang
        # 如果 lang2id 和 id2lang 都不为 None，则它们的长度必须相等
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        # 日语分词器
        self.ja_word_tokenizer = None
        # 中文分词器
        self.zh_word_tokenizer = None

        # 从词汇文件中加载编码器（encoder）
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 从编码器（encoder）中生成解码器（decoder）
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 从合并文件中加载 BPE 合并操作
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        # 将合并操作转换为 BPE 等级字典
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 缓存字典
        self.cache = {}

        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            lang2id=lang2id,
            id2lang=id2lang,
            **kwargs,
        )

    @property
    # 用于获取是否执行小写转换的属性
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.do_lower_case 中复制过来
    def do_lower_case(self):
        return self.do_lowercase_and_remove_accent
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_punct_norm中复制而来的函数
    def moses_punct_norm(self, text, lang):
        # 如果语言不在缓存的moses_punct_normalizer中，则创建一个新的punct_normalizer对象并添加到缓存
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            # 如果语言在缓存中，则直接从缓存中获取punct_normalizer对象
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        # 返回文本经过标点符号规范化后的结果
        return punct_normalizer.normalize(text)
    
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_tokenize中复制而来的函数
    def moses_tokenize(self, text, lang):
        # 如果语言不在缓存的moses_tokenizer中，则创建一个新的moses_tokenizer对象并添加到缓存
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            # 如果语言在缓存中，则直接从缓存中获取moses_tokenizer对象
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        # 返回文本经过分词后的结果
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)
    
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_pipeline中复制而来的函数
    def moses_pipeline(self, text, lang):
        # 替换文本中的unicode标点
        text = replace_unicode_punct(text)
        # 对文本进行标点规范化
        text = self.moses_punct_norm(text, lang)
        # 移除文本中的非打印字符
        text = remove_non_printing_char(text)
        return text
    
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.ja_tokenize中复制而来的函数
    def ja_tokenize(self, text):
        # 如果ja_word_tokenizer为None，则尝试导入Mykytea模块并创建ja_word_tokenizer对象
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea
                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                # 如果导入失败，则输出错误信息并抛出异常
                logger.error("Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper"
                             " (https://github.com/chezou/Mykytea-python) with the following steps")
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                raise
        # 返回文本的分词结果
        return list(self.ja_word_tokenizer.getWS(text))
    
    @property
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.vocab_size中复制而来的属性
    def vocab_size(self):
        # 返回encoder的长度作为vocab_size
        return len(self.encoder)
    
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_vocab中复制而来的函数
    def get_vocab(self):
        # 返回encoder和added_tokens_encoder组成的字典作为词汇表
        return dict(self.encoder, **self.added_tokens_encoder)
    
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.bpe中复制而来的函数
    # 对给定的 token 进行 BPE (Byte Pair Encoding) 处理
    def bpe(self, token):
        # 构造 token 对应的 tuple，最后一个字符添加结束符</w>
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果 token 已经存在于缓存中，则直接返回
        if token in self.cache:
            return self.cache[token]
        # 获取 token 的所有字节对
        pairs = get_pairs(word)

        # 如果没有字节对，则直接返回 token 加上结束符
        if not pairs:
            return token + "</w>"

        # 进行字节对编码
        while True:
            # 找出频率最低的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字节对不在字节对编码的词汇表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 将 word 中所有出现 bigram 的地方替换为新的 bigram
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
            # 如果生成的新的 word 长度为 1，说明编码完成，退出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将 word 转换为字符串形式
        word = " ".join(word)
        # 如果 word 是特殊的换行符形式，则修正为正确的形式
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果存入缓存中
        self.cache[token] = word
        return word

    # 预处理文本，包括替换引号、统一Unicode格式、转换为小写（如果需要）
    def preprocess_text(self, text):
        text = text.replace("``", '"').replace("''", '"')
        text = convert_to_unicode(text)
        text = unicodedata.normalize("NFC", text)

        if self.do_lowercase:
            text = text.lower()

        return text

    # 对文本进行分词，如果bypass_tokenizer为True，则跳过分词直接进行BPE处理
    def _tokenize(self, text, bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        lang = "fr"
        # 如果指定的语言不在已加载的语言映射中，则报错
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by"
                " the loaded pretrained model."
            )

        # 如果 bypass_tokenizer 为 True，则将文本按空格分割为 tokens
        if bypass_tokenizer:
            text = text.split()
        else:
            # 否则先预处理文本，再进行 Moses 分词
            text = self.preprocess_text(text)
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)

        split_tokens = []
        # 对每个 token 进行 BPE 处理
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_token_to_id 复制的函数
    # 将 token（字符串）转换为 ID，如果不在词汇表中则使用 unk_token 对应的 ID
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从索引（整数）转换为 token（字符串）使用词汇表
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_id_to_token 复制而来
    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    # 将一系列的 tokens（字符串）转换为单个字符串
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.convert_tokens_to_string 复制而来
    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    # 为序列分类任务构建模型的输入，通过连接和添加特殊符号
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.build_inputs_with_special_tokens 复制而来
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列或序列对中构建用于序列分类任务的模型输入，通过连接和添加特殊符号。XLM 序列的格式如下：

        - 单个序列：`<s> X </s>`
        - 序列对：`<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                添加特殊符号的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表。

        Returns:
            `List[int]`: 包含适当特殊符号的 [input IDs](../glossary#input-ids) 列表。
        """

    # 获取特殊符号的屏蔽掩码
    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_special_tokens_mask 复制而来
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence
        pair mask has the following format:

        ```py
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
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.save_vocabulary
    # 将词汇表保存到指定目录中，可以添加文件名前缀
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否为有效目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件路径和合并文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 循环写入bpe_tokens到合并文件中
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 从XLMTokenizer的状态中获取数据，用于pickle模块序列化操作
    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    # 设置XLMTokenizer的状态，用于pickle模块反序列化操作
    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__setstate__
    def __setstate__(self, d):
        self.__dict__ = d

        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.sm = sacremoses
```