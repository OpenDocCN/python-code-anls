# `.\transformers\models\mvp\tokenization_mvp_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，使用 Apache 许可证 2.0
import json
from typing import List, Optional, Tuple

# 从 tokenizers 库导入预处理器和处理器
from tokenizers import pre_tokenizers, processors

# 从 tokenization_utils_base 模块中导入 AddedToken 和 BatchEncoding 类
from ...tokenization_utils_base import AddedToken, BatchEncoding
# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 utils 模块中导入 logging 函数
from ...utils import logging
# 从当前目录下的 tokenization_mvp 模块中导入 MvpTokenizer 类
from .tokenization_mvp import MvpTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/vocab.json",
    },
    "added_tokens.json": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/added_tokens.json",
    },
    "merges_file": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/tokenizer.json",
    },
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "RUCAIBox/mvp": 1024,
}

# 定义 MvpTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class MvpTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”MVP分词器（由HuggingFace的*tokenizers*库支持），派生自GPT-2分词器，使用字节级字节对编码。

    这个分词器已经被训练成将空格视为标记的一部分（有点像sentencepiece），因此一个单词将根据它是否在句子的开头（没有空格）等位置而被编码成不同的方式：

    ```python
    >>> from transformers import MvpTokenizerFast

    >>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    通过在实例化此分词器时或在调用一些文本时传递 `add_prefix_space=True`，可以避免这种行为，但由于模型未以此方式进行预训练，这可能会降低性能。

    <Tip>

    当与 `is_split_into_words=True` 一起使用时，需要以 `add_prefix_space=True` 实例化此分词器。

    </Tip>

    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应
    """
    refer to this superclass for more information regarding those methods.

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
            other word. (MVP tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    # Variable for storing names of vocabulary files
    vocab_files_names = VOCAB_FILES_NAMES
    # Variable for storing the mapping of pretrained vocabulary files
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义模型最大输入长度，这些大小是预训练模型使用的
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型需要的输入名称
    model_input_names = ["input_ids", "attention_mask"]
    # 定义慢速 tokenizer 类
    slow_tokenizer_class = MvpTokenizer
    
    # 初始化 MVP Tokenizer 类
    def __init__(
        self,
        # 词汇表文件
        vocab_file=None,
        # 合并操作文件
        merges_file=None,
        # tokenizer 文件
        tokenizer_file=None,
        # 处理错误的策略
        errors="replace",
        # 开始标记
        bos_token="<s>",
        # 结束标记
        eos_token="</s>",
        # 分隔标记
        sep_token="</s>",
        # 分类标记
        cls_token="<s>",
        # 未知标记
        unk_token="<unk>",
        # 填充标记
        pad_token="<pad>",
        # 掩码标记
        mask_token="<mask>",
        # 是否在开头添加空格
        add_prefix_space=False,
        # 是否修剪 offsets
        trim_offsets=True,
        **kwargs,
    ):
        pass
    
    # 获取掩码标记
    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
    
        MVP tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        # 如果没有设置掩码标记，输出错误日志
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回掩码标记
        return str(self._mask_token)
    
    # 设置掩码标记
    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.
    
        This is needed to preserve backward compatibility with all the previously used models based on Mvp.
        """
        # 如果输入的是字符串，则将其转换为 AddedToken 对象，并设置 lstrip 为 True 以保留空格
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value
    
    # 批量编码
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否分词的标志
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果分词但没有设置添加前缀空格，抛出错误
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )
        # 调用父类的批量编码方法
        return super()._batch_encode_plus(*args, **kwargs)
    
    # 单个编码
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否分词的标志
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果分词但没有设置添加前缀空格，抛出错误
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )
        # 调用父类的单个编码方法
        return super()._encode_plus(*args, **kwargs)
    
    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存 tokenizer 模型，返回保存的文件名
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 添加起始标记、输入序列和结束标记，构成最终输出序列
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果不存在第二个输入序列，则直接返回输出序列
        if token_ids_1 is None:
            return output

        # 否则，追加第二个输入序列及结束标记，并返回完整的输出序列
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    # 从输入序列创建特殊标记类型的ids序列
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. MVP does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of zeros.
        """
        # 定义特殊标记（分隔符、类标记）的id列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果不存在第二个输入序列，则返回由0填充的与第一个输入序列长度相同的id列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回由0填充的带有类标记、第一个序列、分隔符、再次分隔符、第二个序列、分隔符的id列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```