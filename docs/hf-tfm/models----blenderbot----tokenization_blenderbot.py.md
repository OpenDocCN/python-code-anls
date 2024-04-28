# `.\transformers\models\blenderbot\tokenization_blenderbot.py`

```
# 设置编码格式为 UTF-8

# 版权声明和许可证信息

# 导入所需的库
import json  # 导入用于处理 JSON 格式的模块
import os  # 导入用于操作系统相关功能的模块
from functools import lru_cache  # 导入用于缓存函数调用结果的装饰器
from typing import List, Optional, Tuple  # 导入用于类型提示的工具

import regex as re  # 导入正则表达式模块

# 导入 Hugging Face 库中的 Tokenizer 和其它实用工具
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入日志记录工具

# 设置日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名的映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "merges_file": "merges.txt",  # 合并规则文件名
    "tokenizer_config_file": "tokenizer_config.json",  # Tokenizer 配置文件名
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"},  # 词汇表文件映射
    "merges_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"},  # 合并规则文件映射
    "tokenizer_config_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"},  # Tokenizer 配置文件映射
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot-3B": 128}  # 预训练模型位置嵌入的尺寸

@lru_cache()  # 使用装饰器 lru_cache 对函数进行结果缓存
# 从 transformers.models.roberta.tokenization_roberta.bytes_to_unicode 复制的函数
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )  # 将 ASCII 可打印字符、特殊字符和扩展 ASCII 字符添加到列表 bs 中
    cs = bs[:]  # 创建列表 cs，其内容与 bs 相同
    n = 0
    for b in range(2**8):  # 遍历 0 到 255 的所有值
        if b not in bs:  # 如果当前值不在列表 bs 中
            bs.append(b)  # 将当前值添加到 bs 中
            cs.append(2**8 + n)  # 将 2^8 + n 添加到 cs 中
            n += 1
    cs = [chr(n) for n in cs]  # 将 cs 中的整数转换为对应的 Unicode 字符
    return dict(zip(bs, cs))  # 返回由 bs 中的值和 cs 中对应位置的 Unicode 字符构成的字典

# 从 transformers.models.roberta.tokenization_roberta.get_pairs 复制的函数
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()  # 创建空集合 pairs
    prev_char = word[0]  # 获取单词的第一个字符
    # 对于单词中除第一个字符外的每个字符
    for char in word[1:]:
        # 将前一个字符和当前字符组成一个元组，加入到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，以便下一次循环使用
        prev_char = char
    # 返回字符对的集合
    return pairs
class BlenderbotTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import BlenderbotTokenizer

    >>> tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
    >>> tokenizer.add_prefix_space = False
    >>> tokenizer("Hello world")["input_ids"]
    [47, 921, 86, 1085, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [6950, 1085, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径
            Path to the vocabulary file.
        merges_file (`str`):
            # 合并文件的路径
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            # 解码字节为 UTF-8 时出现错误时的处理方式
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 在预训练期间用作序列开头的特殊标记，也可用作序列分类器标记
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 序列结束的特殊标记
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符标记，在构建多个序列的序列时使用，例如用于序列分类或用于文本和问题的问答。也用作使用特殊标记构建序列的最后一个标记。
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 分类器标记，在进行序列分类时使用（对整个序列进行分类而不是对每个标记进行分类）。在使用特殊标记构建序列时，它是序列的第一个标记。
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知标记。词汇表中不存在的标记无法转换为 ID，而是设置为此标记。
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 用于填充的标记，例如在批处理不同长度的序列时使用。
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 用于掩码值的标记。在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            # 是否在输入前添加一个初始空格。这允许将前导单词视为任何其他单词。（Blenderbot 分词器通过前导空格检测单词的开头）。
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Blenderbot tokenizer detect beginning of words by the preceding space).
    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入的最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.__init__中复制代码，将Roberta->Blenderbot, RoBERTa->Blenderbot
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 如果bos_token是字符串，则创建一个AddedToken对象，不去除左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果pad_token是字符串，则创建一个AddedToken对象，不去除左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果eos_token是字符串，则创建一个AddedToken对象，不去除左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果unk_token是字符串，则创建一个AddedToken对象，不去除左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果sep_token是字符串，则创建一个AddedToken对象，不去除左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果cls_token是字符串，则创建一个AddedToken对象，不去除左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # mask_token的行为类似于普通单词，即包括它之前的空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 这些特殊标记不是vocab.json的一部分，让我们按正确顺序添加它们
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 如何处理解码中的错误
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # 应该添加re.IGNORECASE，以便可以对缩写的大写版本进行BPE合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.vocab_size中复制代码，将Roberta->Blenderbot, RoBERTa->Blenderbot
    # 返回编码器的长度，即词汇表的大小
    def vocab_size(self):
        return len(self.encoder)

    # 从Blenderbot的词汇表中获取词汇表字典，包括已添加的特殊标记
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 根据Blenderbot的字节对编码（BPE）算法对单词进行编码
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 根据BPE对单词进行拆分并重新组合
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 对文本进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将token转换为对应的ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将ID转换为对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.convert_tokens_to_string复制而来，将Roberta->Blenderbot, RoBERTa->Blenderbot
    def convert_tokens_to_string(self, tokens):
        """将一系列标记（字符串）转换为单个字符串。"""
        # 将标记列表连接成一个字符串
        text = "".join(tokens)
        # 使用字节解码器将字符串转换为UTF-8编码的文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.save_vocabulary复制而来，将Roberta->Blenderbot, RoBERTa->Blenderbot
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"词汇表路径（{save_directory}）应该是一个目录")
            return
        # 构建词汇表文件路径和合并文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器内容写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE标记写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"将词汇表保存到{merge_file}：BPE合并索引不连续。请检查分词器是否损坏！"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_special_tokens_mask复制而来，将Roberta->Blenderbot, RoBERTa->Blenderbot
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        # 如果已经有特殊标记，则调用父类方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有特殊标记，添加特殊标记并返回对应的掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # 从 `transformers.models.roberta.tokenization_roberta.RobertaTokenizer.create_token_type_ids_from_sequences` 复制，并将 RoBERTa 更改为 Blenderbot
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建用于序列对分类任务的掩码。Blenderbot 不使用标记类型 ID，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """
        sep = [self.sep_token_id]  # 分隔符标记的 ID
        cls = [self.cls_token_id]  # 类别标记的 ID

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 返回零列表

    # 从 `transformers.models.roberta.tokenization_roberta.RobertaTokenizer.prepare_for_tokenization` 复制，并将 RoBERTa 更改为 Blenderbot
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)  # 获取是否添加前缀空格的参数
        # 如果文本被拆分成单词或需要添加前缀空格，并且文本长度大于0且第一个字符不是空格，则在文本前添加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)  # 返回处理后的文本和参数
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:
        - single sequence: ` X </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Will be ignored
        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 将给定的 token_ids_0 列表与结束标记的 token ID 进行拼接，以构建输入序列
        return token_ids_0 + [self.eos_token_id]

    @property
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 如果未定义聊天模板，则使用默认模板，并发出警告
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板，其中使用了控制流模板语言
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```