# `.\models\roberta\tokenization_roberta.py`

```
# coding=utf-8
# 版权 2018 年 Open AI 团队作者和 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据"原样"分发，无任何明示或暗示的担保或条件。
# 请参阅许可证获取特定语言的权限。

"""RoBERTa 的分词类。"""

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re  # 导入正则表达式模块

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)  # 获取用于此模块的日志记录器

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",    # 词汇表文件名
    "merges_file": "merges.txt",   # 合并文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {  # 预训练模型的词汇表文件映射
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/vocab.json",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/vocab.json",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/vocab.json",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/vocab.json",
        "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/vocab.json",
        "openai-community/roberta-large-openai-detector": (
            "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/vocab.json"
        ),
    },
    "merges_file": {  # 预训练模型的合并文件映射
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/merges.txt",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/merges.txt",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/merges.txt",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/merges.txt",
        "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/merges.txt",
        "openai-community/roberta-large-openai-detector": (
            "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/merges.txt"
        ),
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "FacebookAI/roberta-base": 512,   # 预训练模型位置嵌入的大小
    "FacebookAI/roberta-large": 512,
    "FacebookAI/roberta-large-mnli": 512,
    "distilbert/distilroberta-base": 512,
    "openai-community/roberta-base-openai-detector": 512,
    # 键："openai-community/roberta-large-openai-detector"，值：512
    "openai-community/roberta-large-openai-detector": 512,
}

@lru_cache()
# 使用 lru_cache 装饰器缓存函数结果，以提高性能，函数无需重复计算相同输入的结果
def bytes_to_unicode():
    """
    返回一个 UTF-8 字节列表，并提供到 Unicode 字符串的映射。
    避免将空白字符和控制字符映射到 BPE（字节对编码）代码无法处理的字符。
    
    可逆的 BPE（字节对编码）在 Unicode 字符串上工作。这意味着如果要避免 UNK（未知）字符，
    则需要在词汇表中包含大量的 Unicode 字符。例如，处理 10B 个标记的数据集时，大约需要 5K 个字符
    来获得良好的覆盖率。这相当于正常情况下 32K 个 BPE 词汇表的显著比例。为了避免这种情况，
    我们需要 UTF-8 字节和 Unicode 字符串之间的查找表。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号元组（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class RobertaTokenizer(PreTrainedTokenizer):
    """
    构建 RoBERTa 分词器，基于 GPT-2 分词器，使用字节级别的字节对编码。

    该分词器已经训练成将空格视为标记的一部分（有点类似 sentencepiece），因此一个单词的编码方式取决于
    它是否在句子开头（没有空格）。

    您可以通过在实例化分词器或在对文本调用时传递 `add_prefix_space=True` 来绕过这种行为，但由于模型
    不是以这种方式进行预训练的，这可能会降低性能。

    <Tip>

    当使用 `is_split_into_words=True` 时，此分词器将在每个单词之前添加一个空格（即使是第一个单词）。

    </Tip>

    此分词器继承自 [`PreTrainedTokenizer`]，该类包含大多数主要方法。用户应参考该超类以获取更多有关这些方法的信息。
    """
    # 定义一个函数签名，说明函数的输入参数和默认值
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇表文件的路径。
        merges_file (`str`):
            Path to the merges file. 合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            解码字节流为 UTF-8 编码时使用的错误处理方式。参见 bytes.decode 获取更多信息。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            训练预处理阶段使用的序列开始标记。可用作序列分类器标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            序列结束标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔符标记，在构建多个序列的序列时使用，例如用于序列分类或文本问答中的问题与回答。同时也作为使用特殊标记构建序列的最后一个标记。

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            用于序列分类任务时的分类器标记（对整个序列进行分类而不是对每个标记进行分类）。在使用特殊标记构建序列时，它是序列的第一个标记。

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。词汇表中不存在的标记无法被转换为标识符，将被设置为此标记。

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时使用。

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            用于掩码值的标记。在进行掩码语言建模训练时使用的标记，模型会试图预测这些标记。

        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
            是否将初始空格添加到输入中。这允许将前导词视为其他任何词。RoBERTa 分词器通过前导空格检测词的开头。
    ```

    # 定义一些常量和列表，用于映射和设置模型输入
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，用于设置特定的tokenizer参数和属性
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
        # 如果bos_token是字符串，则创建一个AddedToken对象，保留左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果pad_token是字符串，则创建一个AddedToken对象，保留左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果eos_token是字符串，则创建一个AddedToken对象，保留左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果unk_token是字符串，则创建一个AddedToken对象，保留左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果sep_token是字符串，则创建一个AddedToken对象，保留左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果cls_token是字符串，则创建一个AddedToken对象，保留左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # Mask token行为类似于普通单词，即在其前面包含空格
        # 如果mask_token是字符串，则创建一个AddedToken对象，去掉左侧空格，保留右侧空格，不进行标准化处理
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 这些特殊标记不包含在vocab.json中，让我们按正确顺序添加它们
        # 使用UTF-8编码打开vocab_file，并加载到self.encoder中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建self.decoder，是self.encoder的反转版本（值-键对）
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码中的错误处理方式
        self.errors = errors  # 如何处理解码中的错误
        # 创建bytes_to_unicode的实例，用于字节到Unicode字符的编码
        self.byte_encoder = bytes_to_unicode()
        # 创建self.byte_decoder，是self.byte_encoder的反转版本（值-键对）
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用UTF-8编码打开merges_file，并按行读取其内容（去掉第一行和最后一行空行）
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将每行的合并操作（merge）拆分为元组，并创建self.bpe_ranks字典
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典
        self.cache = {}
        # 设置是否在特殊标记前添加空格的标志
        self.add_prefix_space = add_prefix_space

        # 应该添加re.IGNORECASE，以便可以对缩写的大写版本进行BPE合并
        # 编译正则表达式模式，用于匹配缩写、字母、数字及其他字符（标点符号等）
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传递参数设置
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

    # vocab_size属性，返回self.encoder字典的长度
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 获取vocab字典，包括self.encoder和added_tokens_encoder的所有内容
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    def _tokenize(self, text):
        """Tokenize a string."""
        # 定义一个空列表，用于存储经过BPE处理后的token
        bpe_tokens = []
        # 使用正则表达式找出文本中的所有匹配项，并遍历每一个token
        for token in re.findall(self.pat, text):
            # 将token转换成UTF-8编码的字节，并逐字节映射到Unicode字符串，避免BPE中的控制标记（在我们的情况下是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 使用BPE算法处理token，并将处理后的子token通过空格拆分并添加到bpe_tokens中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回处理后的token列表
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据token从encoder中获取其对应的id，如果token不存在，则返回未知token的id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据index从decoder中获取其对应的token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将tokens列表中的所有token连接成一个字符串
        text = "".join(tokens)
        # 将字符串转换为UTF-8编码的字节数组，然后解码为Unicode字符串，使用指定的错误处理方式
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回解码后的文本
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构造词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构造合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器（self.encoder）以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并文件，并检查 BPE 合并索引是否连续
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

        # 返回词汇表文件路径和合并文件路径作为元组
        return vocab_file, merge_file


    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

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
        # 如果没有第二个序列（token_ids_1），则返回包含特殊标记的单个序列的输入 IDs
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 构建包含特殊标记的序列对输入 IDs
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep


    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        Retrieve a mask of special tokens to avoid performing unnecessary calculations on them.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs corresponding to the second sequence.
            already_has_special_tokens (`bool`):
                Whether the input token IDs already include special tokens.

        Returns:
            `List[int]`: List indicating the positions of special tokens (1 for special token, 0 otherwise).
        """
        # 初始化一个全零列表，用于记录特殊标记的位置
        special_tokens_mask = [0] * len(token_ids_0)

        # 如果没有第二个序列（token_ids_1），则直接返回全零列表
        if token_ids_1 is None:
            return special_tokens_mask

        # 计算第一个序列的长度
        first_sep_token_idx = token_ids_0.index(self.sep_token_id) if self.sep_token_id in token_ids_0 else -1

        # 遍历第一个序列，将特殊标记的位置设为 1
        for i in range(len(token_ids_0)):
            if token_ids_0[i] in [self.sep_token_id, self.cls_token_id]:
                special_tokens_mask[i] = 1

        # 计算第二个序列的起始位置
        second_sep_token_idx = token_ids_1.index(self.sep_token_id) if self.sep_token_id in token_ids_1 else -1

        # 遍历第二个序列，将特殊标记的位置设为 1
        for i in range(len(token_ids_1)):
            if token_ids_1[i] in [self.sep_token_id, self.cls_token_id]:
                special_tokens_mask.append(1)
            else:
                special_tokens_mask.append(0)

        # 如果输入已经包含特殊标记，将全零列表转换为全一列表
        if already_has_special_tokens:
            special_tokens_mask.extend([1] * (len(token_ids_0) + len(token_ids_1)))

        # 返回特殊标记的掩码列表
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
        # If the token list already has special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there is no token_ids_1 (no second list), return a list with special tokens added to token_ids_0
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # If token_ids_1 exists, return a list with special tokens added to both token_ids_0 and token_ids_1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Initialize special tokens for SEP and CLS
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there is no token_ids_1 (no second list), return a list of zeros of the length of cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If token_ids_1 exists, return a list of zeros of the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepare text for tokenization, potentially adding a prefix space if required.

        Args:
            text (str): The input text to be tokenized.
            is_split_into_words (bool, optional): Whether the text is already split into words.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the modified text and remaining keyword arguments.
        """
        # Determine if a prefix space needs to be added based on conditions
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```