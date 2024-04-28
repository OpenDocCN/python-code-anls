# `.\transformers\models\longformer\tokenization_longformer.py`

```
# coding=utf-8
# 版权声明：本代码由 The Allen Institute for AI team 和 The HuggingFace Inc. team 版权所有
#
# 根据 Apache 许可证 2.0 版本授权；除非符合许可证的要求，否则不得使用此文件。
# 您可以获取许可证副本
# 位于 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依法分发的软件
# 基于“按原样”分布，不提供任何明示或暗示的担保或条件。
# 详细阐释了特定语言下的权限以及对许可证限制的说明
#
# 导入必要的库
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
# 导入 regex 库，并重命名为 re
import regex as re

# 导入 tokenization_utils 模块中的 AddedToken, PreTrainedTokenizer 类
# 导入 utils 模块中的 logging 函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的 logger 对象
logger = logging.get_logger(__name__)

# 定义一个常量字典，包含两个键值对
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 定义一个常量字典，包含多个子字典，每个子字典包含一个模型名称和对应的 URL
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

# 定义一个常量字典，包含多个键值对，键为模型名称，值为位置嵌入的维度大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
    # 键值对："allenai/longformer-large-4096-extra.pos.embd.only"对应值为4096
# 定义一个空行，用于分隔代码段
}

# 使用 lru_cache 装饰器，将下面的函数结果进行缓存，提高函数性能
@lru_cache()
# 从 transformers.models.roberta.tokenization_roberta 模块中复制的 bytes_to_unicode 函数
def bytes_to_unicode():
    """
    返回 utf-8 字节的列表以及到 Unicode 字符串的映射。我们明确避免将字节映射到空白字符/控制字符，因为 bpe 代码会因此而出错。

    可逆的 bpe 代码适用于 Unicode 字符串。这意味着如果要避免 UNK（未知标记），则需要在词汇表中包含大量的 Unicode 字符。当您处理约 100 亿个令牌的数据集时，您最终需要大约 5000 个字符来获得良好的覆盖率。这在常规的 32K bpe 词汇表中占据了相当大的比例。为了避免这种情况，我们希望在 utf-8 字节和 Unicode 字符串之间建立查找表。
    """
    # 定义 utf-8 字节列表，包括常用的 ASCII 字符范围，以及扩展字符范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 复制 utf-8 字节列表以创建字节映射
    cs = bs[:]
    n = 0
    # 添加不在 bs 列表中的字节到 bs 列表，并为它们分配编码
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 将编码转换为对应的 Unicode 字符
    cs = [chr(n) for n in cs]
    # 返回字节到 Unicode 字符的映射字典
    return dict(zip(bs, cs))


# 从 transformers.models.roberta.tokenization_roberta 模块中复制的 get_pairs 函数
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号的元组（符号是长度可变的字符串）。
    """
    # 初始化符号对集合
    pairs = set()
    prev_char = word[0]
    # 遍历单词中的每个符号，将相邻符号组成的对添加到集合中
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    # 返回符号对集合
    return pairs


# 从 transformers.models.roberta.tokenization_roberta 模块中复制的 LongformerTokenizer 类
class LongformerTokenizer(PreTrainedTokenizer):
    """
    构建 Longformer 分词器，衍生自 GPT-2 分词器，使用字节级别的 BPE（Byte-Pair-Encoding）。

    该分词器已经被训练，以将空格视为标记的一部分（有点像 sentencepiece），因此一个单词的编码方式会因其是否在句子开头而不同（没有空格或有空格）：

    您可以通过在实例化分词器时或在调用文本时传递 `add_prefix_space=True` 来避免这种行为，但由于模型不是这种方式进行预训练的，这可能会导致性能下降。

    <提示>

    当与 `is_split_into_words=True` 一起使用时，该分词器将在每个单词之前添加一个空格（即使是第一个单词）。

    </提示>

    该分词器继承自 [`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Longformer tokenizer detect beginning of words by the preceding space).
    """

    # 定义常量，用于指定词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练位置嵌入的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化方法用于设置Tokenizer对象的特殊标记和参数
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
        # 处理特殊标记，如果是字符串类型则创建AddedToken对象，否则使用已有对象
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # 处理mask_token，使其像普通单词一样，包含之前的空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 读取vocab_file中的json数据作为encoder字典
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        # 根据encoder字典生成decoder字典
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 解码中如何处理错误
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 读取merges_file处理成bpe_merges列表，并用字典生成bpe_merges对应的序号字典
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # 创建正则表达式模式，用于识别特定格式的单词
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传递参数进行初始化
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

    # vocab_size属性用于返回encoder字典的长度
    @property
    def vocab_size(self):
        return len(self.encoder)

    # get_vocab方法返回包含encoder和added_tokens_encoder的复制字典
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    # 对输入的 token 进行 BPE 编码处理
    def bpe(self, token):
        # 如果 token 已经存在于缓存中，则直接返回
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换成元组形式
        word = tuple(token)
        # 获取 token 的所有 pairs
        pairs = get_pairs(word)

        # 如果 pairs 为空，则直接返回 token
        if not pairs:
            return token

        # 循环处理 token，直到无法再生成新的 bigram 为止
        while True:
            # 找到当前 pairs 中 bpe_ranks 最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果 bigram 不在 bpe_ranks 中，则退出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 循环处理 word 中的字符，根据 bigram 进行合并
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
            # 将处理后的 new_word 转换成元组，用于下一个循环
            new_word = tuple(new_word)
            word = new_word
            # 如果 word 中只剩下一个字符，则退出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将处理后的 word 转换成字符串形式
        word = " ".join(word)
        # 将处理后的 token 存入缓存中
        self.cache[token] = word
        return word

    # 将文本进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 对 token 进行编码，避免 BPE 的控制符号
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将编码后的 token 进行 BPE 处理，然后拆分成词
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将 token 转换成对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换成对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列 token 转换成字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        # 将 token 转换成可读的字符串形式
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则返回错误信息
        if not os.path.isdir(save_directory):
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

        # 将 encoder 对象以 JSON 格式写入到词汇表文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引
        index = 0
        # 将 BPE tokens 和对应的索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    # 为序列构建特殊标记的输入
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
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果仅有一个序列，则在序列开头和结尾分别添加特殊标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 创建特殊标记列表并返回
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 获取特殊标记的屏蔽掩码
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
        # 如果已经有特殊token，直接调用基类方法返回特殊token掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有特殊token，根据情况生成特殊token掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
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
        # 创建用于序列对分类任务的掩码，Longformer模型不使用token类型ID，因此返回全为0的列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 根据参数设置文本格式化，确保符合特定要求
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```  
```