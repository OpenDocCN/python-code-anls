# `.\models\longformer\tokenization_longformer.py`

```
# 导入所需模块和库
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统相关功能的模块
from functools import lru_cache  # 导入用于缓存函数调用结果的装饰器
from typing import List, Optional, Tuple  # 导入用于类型提示的模块

import regex as re  # 导入正则表达式库，命名为 re

# 从 tokenization_utils 模块中导入 AddedToken 和 PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 导入日志记录模块中的日志记录器
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/vocab.json"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/vocab.json"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/vocab.json"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/merges.txt"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/merges.txt"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/merges.txt"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/merges.txt"
        ),
    },
}

# 预训练位置嵌入的尺寸映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
}
    # 定义一个字符串键值对，键是文件路径，值是整数 4096
    "allenai/longformer-large-4096-extra.pos.embd.only": 4096,
}


@lru_cache()
# 从transformers.models.roberta.tokenization_roberta.bytes_to_unicode中复制而来
# 返回一个字节到Unicode字符串的映射表
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    # 定义一个字节列表bs，包含了utf-8编码中可打印字符和特定范围内的其他字符
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 复制一份字节列表到cs
    cs = bs[:]
    # 初始化一个计数器n为0
    n = 0
    # 遍历0到255之间的所有字节
    for b in range(2**8):
        # 如果当前字节b不在bs列表中
        if b not in bs:
            # 将b添加到bs列表中
            bs.append(b)
            # 将2**8 + n添加到cs列表中，并增加计数器n
            cs.append(2**8 + n)
            n += 1
    # 将cs列表中的每个整数转换为对应的Unicode字符，形成一个字节到Unicode字符的映射表并返回
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# 从transformers.models.roberta.tokenization_roberta.get_pairs中复制而来
# 返回一个单词中的符号对的集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 初始化一个空集合pairs
    pairs = set()
    # 从单词的第一个符号开始遍历到倒数第二个符号
    prev_char = word[0]
    for char in word[1:]:
        # 将相邻的符号对(prev_char, char)加入到pairs集合中
        pairs.add((prev_char, char))
        prev_char = char
    # 返回符号对的集合pairs
    return pairs


# 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer中复制而来
class LongformerTokenizer(PreTrainedTokenizer):
    """
    Constructs a Longformer tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import LongformerTokenizer

    >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 构造一个Longformer分词器，继承自PreTrainedTokenizer类

    def __init__(self, *init_inputs, **kwargs):
        # 调用父类的构造函数，传入所有初始化参数和关键字参数
        super().__init__(*init_inputs, **kwargs)

    # LongformerTokenizer类还有其他方法和属性，在此省略...
    # 定义一个名为 vocab_file 的参数，表示词汇表文件的路径
    vocab_file (`str`):
        Path to the vocabulary file.
    # 定义一个名为 merges_file 的参数，表示合并文件的路径
    merges_file (`str`):
        Path to the merges file.
    # 定义一个名为 errors 的参数，表示解码字节为 UTF-8 时的错误处理方式，默认为 "replace"
    errors (`str`, *optional*, defaults to `"replace"`):
        Paradigm to follow when decoding bytes to UTF-8. See
        [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
    # 定义一个名为 bos_token 的参数，表示序列的起始标记，默认为 `"<s>"`
    bos_token (`str`, *optional*, defaults to `"<s>"`):
        The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        <Tip>

        When building a sequence using special tokens, this is not the token that is used for the beginning of
        sequence. The token used is the `cls_token`.

        </Tip>

    # 定义一个名为 eos_token 的参数，表示序列的结束标记，默认为 `"</s>"`
    eos_token (`str`, *optional*, defaults to `"</s>"`):
        The end of sequence token.

        <Tip>

        When building a sequence using special tokens, this is not the token that is used for the end of sequence.
        The token used is the `sep_token`.

        </Tip>

    # 定义一个名为 sep_token 的参数，表示序列的分隔标记，默认为 `"</s>"`
    sep_token (`str`, *optional*, defaults to `"</s>"`):
        The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
        sequence classification or for a text and a question for question answering. It is also used as the last
        token of a sequence built with special tokens.
    # 定义一个名为 cls_token 的参数，表示分类器标记，默认为 `"<s>"`
    cls_token (`str`, *optional*, defaults to `"<s>"`):
        The classifier token which is used when doing sequence classification (classification of the whole sequence
        instead of per-token classification). It is the first token of the sequence when built with special tokens.
    # 定义一个名为 unk_token 的参数，表示未知标记，默认为 `"<unk>"`
    unk_token (`str`, *optional*, defaults to `"<unk>"`):
        The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
        token instead.
    # 定义一个名为 pad_token 的参数，表示填充标记，默认为 `"<pad>"`
    pad_token (`str`, *optional*, defaults to `"<pad>"`):
        The token used for padding, for example when batching sequences of different lengths.
    # 定义一个名为 mask_token 的参数，表示掩码标记，默认为 `"<mask>"`
    mask_token (`str`, *optional*, defaults to `"<mask>"`):
        The token used for masking values. This is the token used when training this model with masked language
        modeling. This is the token which the model will try to predict.
    # 定义一个名为 add_prefix_space 的参数，表示是否在输入开头添加空格，默认为 `False`
    add_prefix_space (`bool`, *optional*, defaults to `False`):
        Whether or not to add an initial space to the input. This allows to treat the leading word just as any
        other word. (Longformer tokenizer detect beginning of words by the preceding space).

vocab_files_names = VOCAB_FILES_NAMES
pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
model_input_names = ["input_ids", "attention_mask"]
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
        # 如果 `bos_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 `pad_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果 `eos_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 `unk_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 `sep_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果 `cls_token` 是字符串，则创建一个 `AddedToken` 对象，保留其左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # `mask_token` 被视为普通单词，即在其前面包含空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 这些特殊标记不包含在 `vocab.json` 中，让我们按正确的顺序添加它们
        # 使用 UTF-8 编码打开 `vocab_file`，加载其中的 JSON 数据到 `self.encoder`
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建一个反向映射，将 `self.encoder` 的键值对调，存储到 `self.decoder`
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式为 `errors`
        self.errors = errors  # how to handle errors in decoding
        # 创建字节到 Unicode 的编码映射
        self.byte_encoder = bytes_to_unicode()
        # 创建一个反向映射，将 `self.byte_encoder` 的键值对调，存储到 `self.byte_decoder`
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 UTF-8 编码打开 `merges_file`，读取并分割为 BPE 合并列表 `bpe_merges`
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将每个 BPE 合并规则字符串转换为元组，并创建其对应的索引字典 `self.bpe_ranks`
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典
        self.cache = {}
        # 是否在添加前缀空格
        self.add_prefix_space = add_prefix_space

        # 应该添加 `re.IGNORECASE` 以便对缩写的大写版本进行 BPE 合并
        # 编译正则表达式 `self.pat`，匹配 `'s`、`'t`、`'re`、`'ve`、`'m`、`'ll`、`'d`、`\p{L}+`、`\p{N}+`、`[^\s\p{L}\p{N}]+`、不跟随非空白字符的空格
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传递参数和关键字参数
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
    def vocab_size(self):
        # 返回 `self.encoder` 中键的数量，即词汇表的大小
        return len(self.encoder)

    def get_vocab(self):
        # 创建 `vocab` 字典，复制 `self.encoder` 的内容，然后更新添加的特殊标记编码映射
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    def bpe(self, token):
        # 如果 token 已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 获取 token 的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回原始 token
        if not pairs:
            return token

        # 反复处理字符对，直到无法继续拆分
        while True:
            # 找到当前权重最小的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的字符对不在预先计算的排名中，则停止拆分
            if bigram not in self.bpe_ranks:
                break
            # 分离出字符对的两个部分
            first, second = bigram
            new_word = []
            i = 0
            # 遍历当前 word 中的字符
            while i < len(word):
                try:
                    # 查找字符对的第一个字符在 word 中的位置
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到，则将剩余部分直接加入新单词中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将非字符对部分加入新单词中
                    new_word.extend(word[i:j])
                    i = j

                # 检查当前位置是否匹配字符对的第一个和第二个字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 如果匹配，将字符对作为一个单元添加到新单词中
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符添加到新单词中，并移动到下一个位置
                    new_word.append(word[i])
                    i += 1
            # 将新单词转换为元组形式，更新 word 变量
            new_word = tuple(new_word)
            word = new_word
            # 如果新单词长度为 1，则停止拆分
            if len(word) == 1:
                break
            else:
                # 继续获取新的字符对
                pairs = get_pairs(word)
        
        # 将拆分后的单词连接成字符串形式
        word = " ".join(word)
        # 将结果缓存起来，避免重复计算
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化空的 BPE tokens 列表
        bpe_tokens = []
        # 使用正则表达式找出所有符合条件的 token
        for token in re.findall(self.pat, text):
            # 将每个 token 转换为 BPE token，并加入到 bpe_tokens 中
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 unicode 字符串，避免 BPE 中的控制符（在我们的情况下是空格）
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将 token 转换为对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据词汇表将 id 转换为对应的 token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列 token 连接成一个字符串
        text = "".join(tokens)
        # 将字符串中的每个字节解码为 unicode 字符串，使用指定的错误处理方法
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇文件路径，结合保存目录和文件名前缀（如果有的话）
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径，结合保存目录和文件名前缀（如果有的话）
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器（self.encoder）的内容以 JSON 格式写入词汇文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE 合并信息写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 对 BPE 合并信息按照索引排序并写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果 BPE 合并索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇文件路径和合并文件路径
        return vocab_file, merge_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Longformer sequence has the following format:

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
        # 如果只有一个输入序列，添加起始和结束特殊标记，并返回结果
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # 如果有两个输入序列，添加起始、分隔、分隔以及第二个序列的起始和结束特殊标记，并返回结果
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        # 确定输入序列是否已经包含了特殊标记
        # 如果已经包含了特殊标记，则创建一个与输入序列长度相同的掩码，所有特殊标记位置为1，其余为0
        if already_has_special_tokens:
            return [1] * len(token_ids_0)
        
        # 初始化一个空列表作为掩码
        special_tokens_mask = []
        # 遍历第一个输入序列的每个元素，将特殊标记位置设为1，其余为0
        for token_id in token_ids_0:
            special_tokens_mask.append(1 if token_id in [self.cls_token_id, self.sep_token_id] else 0)
        
        # 如果有第二个输入序列，同样处理它的特殊标记
        if token_ids_1 is not None:
            for token_id in token_ids_1:
                special_tokens_mask.append(1 if token_id in [self.cls_token_id, self.sep_token_id] else 0)
        
        # 返回最终生成的特殊标记掩码
        return special_tokens_mask
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
        # 如果已经有特殊的标记，直接调用父类方法获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有特殊的标记，根据输入的 token_ids_1 是否为 None，决定返回的特殊标记的掩码列表
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 初始化 SEP 和 CLS 标记的列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 根据 token_ids_1 是否为 None，返回相应长度的全零列表作为 token type ids
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        else:
            return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepare text for tokenization by optionally adding a space prefix based on the arguments.

        Args:
            text (str): Input text to be tokenized.
            is_split_into_words (bool, optional): Whether the text is already split into words.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[str, dict]: Processed text and any remaining keyword arguments.
        """
        # 获取 add_prefix_space 参数，默认使用对象的设置
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        
        # 如果需要添加前缀空格，并且文本不以空格开头，则在文本前添加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        
        # 返回处理后的文本和可能修改过的关键字参数
        return (text, kwargs)
```