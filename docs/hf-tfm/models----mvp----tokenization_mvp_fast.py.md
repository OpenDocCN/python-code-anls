# `.\models\mvp\tokenization_mvp_fast.py`

```py
import json  # 导入处理 JSON 数据的模块
from typing import List, Optional, Tuple  # 引入类型提示相关的模块

from tokenizers import pre_tokenizers, processors  # 导入 tokenizers 库中的预处理器和处理器

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 导入基础的 tokenization_utils_base 模块中的类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...utils import logging  # 导入 logging 模块中的 logging 函数
from .tokenization_mvp import MvpTokenizer  # 从当前目录下的 tokenization_mvp 模块导入 MvpTokenizer 类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}
# 定义包含各种文件名的字典，用于 MVP tokenizer 的词汇文件和相关文件

# 预训练模型的文件映射，指定每个预训练模型的相关文件的 URL
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

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "RUCAIBox/mvp": 1024,  # 预训练模型 RUCAIBox/mvp 的位置嵌入大小为 1024
}


class MvpTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" MVP tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2 tokenizer,
    using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import MvpTokenizerFast

    >>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    ```
    # MvpTokenizerFast 类，是基于 HuggingFace 的 tokenizers 库实现的 MVP tokenizer，使用字节级别的 BPE 进行编码
    # 继承自 PreTrainedTokenizerFast 类，包含大多数主要方法，支持从预训练模型加载和使用
    # 通过示例展示了该 tokenizer 如何处理空格以及在不同位置编码单词的不同方式
    # 提示用户在特定情况下实例化时需要传递额外参数以获得最佳效果
    # 定义一个类，用于实现基于特定语言模型的Tokenizer，继承自一个提供了相关方法信息的超类。
    class PreTrainedTokenizer(PreTrainedTokenizerBase):
        # 初始化方法，接受词汇表文件路径、合并文件路径和可选的解码错误处理方式等参数。
        def __init__(
            self,
            vocab_file: str,
            merges_file: str,
            errors: str = "replace",
            bos_token: str = "<s>",
            eos_token: str = "</s>",
            sep_token: str = "</s>",
            cls_token: str = "<s>",
            unk_token: str = "<unk>",
            pad_token: str = "<pad>",
            mask_token: str = "<mask>",
            add_prefix_space: bool = False,
            trim_offsets: bool = True,
        ):
            # 调用超类的初始化方法，传递必要的参数
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            )
            # 记录词汇表文件名和预训练词汇表文件映射
            self.vocab_files_names = vocab_files_names
            self.pretrained_vocab_files_map = pretrained_vocab_files_map
    # 将预训练模型的位置编码大小赋给 max_model_input_sizes 变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速分词器的类别为 MvpTokenizer
    slow_tokenizer_class = MvpTokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        # 初始化方法，设置各种初始化参数

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        MVP tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        # 获取 mask_token 属性的 getter 方法，返回当前的 mask token 字符串
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on Mvp.
        """
        # 设置 mask_token 属性的 setter 方法，使其吞噬前面的空格
        # 如果 value 是字符串，则创建一个 AddedToken 对象，设置 lstrip 为 True，rstrip 为 False
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 批量编码方法，返回 BatchEncoding 对象
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            # 如果输入已经被分割成单词但没有加前缀空格，则抛出错误
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 编码方法，返回 BatchEncoding 对象
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            # 如果输入已经被分割成单词但没有加前缀空格，则抛出错误
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存词汇表到指定目录
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
    # 构建包含特殊标记的输入序列，用于模型输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 初始化输出列表，以起始标记开始，接着是token_ids_0，最后是结束标记
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果没有第二个序列token_ids_1，则直接返回构建好的output
        if token_ids_1 is None:
            return output
        
        # 如果有第二个序列token_ids_1，则在output后面添加结束标记，接着是token_ids_1，最后再加一个结束标记
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    # 根据输入的两个序列token_ids_0和token_ids_1创建token type ids序列
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
            `List[int]`: List of zeros.
        """
        # 分隔符标记列表
        sep = [self.sep_token_id]
        # 分类起始标记列表
        cls = [self.cls_token_id]

        # 如果没有第二个序列token_ids_1，则返回一个全为0的列表，长度为cls + token_ids_0 + sep的长度
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # 如果有第二个序列token_ids_1，则返回一个全为0的列表，长度为cls + token_ids_0 + sep + sep + token_ids_1 + sep的长度
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```