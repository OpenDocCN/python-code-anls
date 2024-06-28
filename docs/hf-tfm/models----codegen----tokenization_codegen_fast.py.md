# `.\models\codegen\tokenization_codegen_fast.py`

```
# 导入所需的模块和库
import json  # 导入处理 JSON 格式数据的模块
import re  # 导入正则表达式模块，用于文本处理
from typing import TYPE_CHECKING, List, Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入处理数组数据的 NumPy 库

# 导入日志记录模块
from ...utils import is_tf_available, is_torch_available, logging

# 检查类型注解，确定是否导入 torch 或 tensorflow 相关模块
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf

# 导入 tokenizers 库中的预处理模块
from tokenizers import pre_tokenizers

# 导入基础的 tokenization_utils_base 模块中的 BatchEncoding 类
from ...tokenization_utils_base import BatchEncoding

# 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast

# 导入本地的 tokenization_codegen 模块中的 CodeGenTokenizer 类
from .tokenization_codegen import CodeGenTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "Salesforce/codegen-350M-mono": (
            "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练模型的位置编码大小映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}


class CodeGenTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”CodeGen分词器（由HuggingFace的*tokenizers*库支持）。基于字节级的 Byte-Pair-Encoding。

    这个分词器经过训练，将空格视为标记的一部分（类似于sentencepiece），因此一个单词的编码方式会因其是否位于句子开头而不同（没有空格或有空格）：

    ```python
    >>> from transformers import CodeGenTokenizerFast

    >>> tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    如果在实例化分词器时传入 `add_prefix_space=True`，可以避免这种行为，但由于模型未以这种方式进行预训练，可能会降低性能。

    <Tip>

    当 `is_split_into_words=True` 时，需要使用 `add_prefix_space=True` 实例化这个分词器。
    """
    # 定义类 CodeGenTokenizer，继承自 PreTrainedTokenizerFast 类，包含大多数主要方法
    This tokenizer inherits from `PreTrainedTokenizerFast` which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    # 初始化方法，接受多个参数来配置 tokenizer
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
            词汇表文件的路径。
        merges_file (`str`, *optional*):
            Path to the merges file.
            merges 文件的路径。
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
            tokenizers 文件的路径，通常为 .json 扩展名，包含加载 tokenizer 所需的全部信息。
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。不在词汇表中的标记无法转换为 ID，并将设置为此标记。
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
            序列的开始标记。
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
            序列的结束标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (CodeGen tokenizer detect beginning of words by the preceding space).
            是否在输入前添加一个初始空格。这样可以将前导单词视为任何其他单词。（CodeGen tokenizer 通过前导空格检测单词的开始）。
    """

    # 定义常量，指定词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速 tokenizer 的类为 CodeGenTokenizer
    slow_tokenizer_class = CodeGenTokenizer

    # 初始化方法
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
    ):
        # 调用父类的构造函数，初始化一个新的实例
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 如果在参数 kwargs 中设置了 "add_bos_token"，则抛出错误
        if kwargs.pop("add_bos_token", False):
            model_id = kwargs.pop("name_or_path", "")
            raise ValueError(
                "Currenty GPT2's fast tokenizer does NOT support adding a BOS token. "
                "Instead you should use GPT2's slow tokenizer class `CodeGenTokenizer` as follows: \n"
                f"`CodeGenTokenizer.from_pretrained('{model_id}')`\nor\n"
                f"`AutoTokenizer.from_pretrained('{model_id}', use_fast=False)`\n"
                "This issue will be fixed soon, see: https://github.com/huggingface/tokenizers/pull/1005."
                " so that the fast tokenizer works correctly."
            )

        # 获取当前预处理器的状态并将其转换为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果预处理器的 "add_prefix_space" 参数与当前实例中的 add_prefix_space 不一致，则更新预处理器的状态
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置实例的 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

    # 重写父类的 _batch_encode_plus 方法，返回 BatchEncoding 对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否已经将输入拆分为单词的参数，默认为 False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 断言如果 add_prefix_space 为 True 或者未将输入拆分为单词，则抛出错误
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的 _batch_encode_plus 方法并返回结果
        return super()._batch_encode_plus(*args, **kwargs)

    # 重写父类的 _encode_plus 方法，返回 BatchEncoding 对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否已经将输入拆分为单词的参数，默认为 False
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 断言如果 add_prefix_space 为 True 或者未将输入拆分为单词，则抛出错误
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的 _encode_plus 方法并返回结果
        return super()._encode_plus(*args, **kwargs)

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 Tokenizer 的 model.save 方法保存模型文件到指定目录，并返回文件名的元组
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    # 解码 token_ids 到原始文本
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """

        # 使用继承自父类的方法 `decode` 对 token_ids 进行解码
        decoded_text = super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        # 如果指定了 `truncate_before_pattern`，则根据正则表达式列表进行截断
        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        # 返回解码后的文本
        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        # 内部函数，用于在字符串中查找正则表达式的位置
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        # 编译正则表达式列表为多行模式的正则对象
        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        # 查找代码字符串中以 "^print" 开头的所有位置
        prints = list(re.finditer("^print", completion, re.MULTILINE))

        # 如果找到多于一个 "^print" 开头的位置，则截断字符串到第二个 "^print" 之前
        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        # 查找代码字符串中以 "^def" 开头的所有位置
        defs = list(re.finditer("^def", completion, re.MULTILINE))

        # 如果找到多于一个 "^def" 开头的位置，则截断字符串到第二个 "^def" 之前
        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        # 查找代码字符串中所有 `truncate_before_pattern` 匹配的位置
        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        # 如果找到任何一个 `truncate_before_pattern` 的位置，则截断字符串到最小的位置处
        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion
```