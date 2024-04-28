# `.\models\deberta\tokenization_deberta_fast.py`

```
# 设定文件编码为utf-8
# 版权声明，告知文件使用者遵守Apache License, Version 2.0
# 获取Apache License, Version 2.0的副本链接
# 如果适用法律要求或以书面形式同意，按“原样”基础分发的软件，无论有无保证或条件
# 请查看特定语言的许可证以获取权限和限制
""" 为DeBERTa模型提供快速分词类。"""

# 引入必要的模块和类型
import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers

from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_deberta import DebertaTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义用于DeBERTa模型的词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/merges.txt"
        ),
    },
}

# 预训练的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
}

# 预训练的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    # 将 "microsoft/deberta-large" 作为键，False 作为值，添加到字典中
# 这是一个 DebertaTokenizerFast 类的定义，该类是一个基于 byte-level Byte-Pair-Encoding 的快速 DeBERTa 分词器
class DebertaTokenizerFast(PreTrainedTokenizerFast):
    """
    # 构建一个基于 HuggingFace 的 tokenizers 库的"快速"DeBERTa 分词器。
    # 该分词器被训练为将空格视为标记的一部分(类似于 sentencepiece)，因此单词的编码方式会因为它在句子开头是否有空格而有所不同。
    Construct a "fast" DeBERTa tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    # 这个示例展示了这种行为:
    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    # 创建一个 DebertaTokenizerFast 实例
    >>> from transformers import DebertaTokenizerFast
    >>> tokenizer = DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")
    
    # 对"Hello world"进行编码,得到的 input_ids 为[1, 31414, 232, 2]
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    # 对" Hello world"进行编码,得到的 input_ids 为[1, 20920, 232, 2]
    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
    ```

    # 可以通过在实例化该分词器时传入 `add_prefix_space=True` 来避免这种行为,但由于模型并没有以这种方式预训练,可能会导致性能下降。

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    # 提示:当与 `is_split_into_words=True` 一起使用时,这个分词器需要用 `add_prefix_space=True` 实例化。

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    # 这个分词器继承自 `PreTrainedTokenizerFast`,包含了大部分主要方法。用户应该参考该父类以获取更多信息。

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 定义一个类 DebertaTokenizer
    class DebertaTokenizer:
        # 初始化方法，接收词汇文件、合并文件、标记器文件和其他参数
        def __init__(
            self,
            vocab_file=None,           # 词汇文件路径
            merges_file=None,           # 合并文件路径
            tokenizer_file=None,        # 标记器文件路径
            errors="replace",           # 解码字节为 UTF-8 时出现错误的处理方式
            bos_token="[CLS]",          # 序列的开头标记
            eos_token="[SEP]",          # 序列的结尾标记
            sep_token="[SEP]",          # 用于建立多个序列的分隔符
            cls_token="[CLS]",          # 在进行序列分类时使用的分类器标记
            unk_token="[UNK]",          # 未知标记
            pad_token="[PAD]",          # 用于填充的标记
            mask_token="[MASK]",        # 用于掩盖值的标记
            add_prefix_space=False,     # 是否在输入的开头添加一个空格
            **kwargs,                   # 其他关键字参数
    ):
        # 调用父类的构造函数，初始化 DebertaTokenizer
        super().__init__(
            vocab_file,  # 词汇表文件路径
            merges_file,  # 合并文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            errors=errors,  # 错误处理方式
            bos_token=bos_token,  # 句子开头 token
            eos_token=eos_token,  # 句子结尾 token
            unk_token=unk_token,  # 未知 token
            sep_token=sep_token,  # 分隔 token
            cls_token=cls_token,  # 分类 token
            pad_token=pad_token,  # 填充 token
            mask_token=mask_token,  # 掩码 token
            add_prefix_space=add_prefix_space,  # 是否在 token 前加空格
            **kwargs,  # 其他参数
        )
        # 检查是否要添加句子开头 token
        self.add_bos_token = kwargs.pop("add_bos_token", False)

        # 获取前置分词器状态
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果前置分词器是否在 token 前加空格与指定值不同，则更新其状态
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 保存是否在 token 前加空格的设置
        self.add_prefix_space = add_prefix_space

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        Deberta tokenizer has a special mask token to be used in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *[MASK]*.
        """
        # 如果未设置掩码 token，则记录错误信息并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.
        """
        # 使掩码 token 像普通单词一样，包括前面的空格
        # 因此，设置 lstrip 为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

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
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._batch_encode_plus
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._encode_plus
    # 定义一个方法用于编码文本序列，返回一个BatchEncoding对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取关键字参数"is_split_into_words"的值，默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 断言是否添加了前缀空格，如果is_split_into_words为True，则需要添加前缀空格，否则抛出异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的_encode_plus方法，并返回结果
        return super()._encode_plus(*args, **kwargs)

    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast中复制了该方法
    # 用于保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用底层tokenizer模型的save方法保存词汇表到指定目录，返回保存的文件路径
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 将文件路径打包成元组返回
        return tuple(files)
```