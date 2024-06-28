# `.\models\lxmert\tokenization_lxmert_fast.py`

```py
# coding=utf-8
# 引入必要的库和模块
import json  # 导入 json 库，用于处理 JSON 数据
from typing import List, Optional, Tuple  # 导入类型提示模块，用于类型标注

from tokenizers import normalizers  # 从 tokenizers 库中导入 normalizers 模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的 tokenizer 类
from .tokenization_lxmert import LxmertTokenizer  # 从当前目录导入 LxmertTokenizer 类

# 定义与词汇相关的文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "unc-nlp/lxmert-base-uncased": (
            "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unc-nlp/lxmert-base-uncased": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "unc-nlp/lxmert-base-uncased": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制而来，修改为 Lxmert 相关的类和文件名
class LxmertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Lxmert tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 引入全局变量，包含预定义的词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    
    # 引入预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    
    # 引入预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    
    # 引入预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 引入慢速分词器的类，这里指定为 LxmertTokenizer
    slow_tokenizer_class = LxmertTokenizer
    
    # 初始化方法，用于创建一个新的实例
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，可选参数
        tokenizer_file=None,  # 分词器文件路径，可选参数
        do_lower_case=True,  # 是否将输入转换为小写，可选参数，默认为 True
        unk_token="[UNK]",  # 未知标记，词汇表中不存在的标记
        sep_token="[SEP]",  # 分隔符标记，用于构建多个序列的序列
        pad_token="[PAD]",  # 填充标记，用于将不同长度的序列填充到相同长度
        cls_token="[CLS]",  # 分类器标记，在序列分类时作为序列的第一个标记
        mask_token="[MASK]",  # 掩码标记，用于掩码语言建模中的预测
        tokenize_chinese_chars=True,  # 是否对中文字符进行分词，可选参数，默认为 True
        strip_accents=None,  # 是否去除所有的重音符号，如果未指定，则由 lowercase 参数决定
        **kwargs,  # 其他参数，以字典形式接收
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

# 调用父类构造函数，并传入初始化参数，包括词汇文件路径、分词器文件路径、大小写转换标志、未知标记、分隔标记、填充标记、类标记、掩码标记、处理中文字符的标志、去除重音符号的标志以及其他关键字参数。


        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

# 加载后端分词器的标准化器状态，并根据初始化参数检查是否需要更新标准化器的设置（如小写转换、去除重音符号、处理中文字符）。如果有变化，则更新标准化器的类和相关参数设置。


        self.do_lower_case = do_lower_case

# 将初始化参数中的 `do_lower_case` 值保存到实例变量 `self.do_lower_case` 中。


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Lxmert sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

# 构建模型输入，根据序列或序列对的情况连接并添加特殊标记。Lxmert 序列有特定的格式：单个序列包括 `[CLS] X [SEP]`，序列对包括 `[CLS] A [SEP] B [SEP]`。函数接受两个参数 `token_ids_0` 和 `token_ids_1`，分别是待添加特殊标记的ID列表，返回一个列表，包含输入ID及相应的特殊标记。


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None

# 创建用于区分两个序列的 token type IDs 的方法。接受两个参数 `token_ids_0` 和 `token_ids_1`，分别是序列的ID列表，返回一个标识序列类型的ID列表。```
):
    super().__init__(
        vocab_file,
        tokenizer_file=tokenizer_file,
        do_lower_case=do_lower_case,
        unk_token=unk_token,
        sep_token=sep_token,
        pad_token=pad_token,
        cls_token=cls_token,
        mask_token=mask_token,
        tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents,
        **kwargs,
    )

# 调用父类的构造函数，并传入初始化参数，包括词汇文件路径、分词器文件路径、大小写转换标志、未知标记、分隔标记、填充标记、类标记、掩码标记、处理中文字符的标志，以及其他关键字参数。


    normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
    if (
        normalizer_state.get("lowercase", do_lower_case) != do_lower_case
        or normalizer_state.get("strip_accents", strip_accents) != strip_accents
        or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
    ):
        normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
        normalizer_state["lowercase"] = do_lower_case
        normalizer_state["strip_accents"] = strip_accents
        normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
        self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

# 加载后端分词器的标准化器状态，并检查是否需要更新标准化器的设置（如小写转换、去除重音符号、处理中文字符）。如果有变化，则通过反射获取标准化器类，并更新相关参数后重新设置到 `self.backend_tokenizer.normalizer` 中。


    self.do_lower_case = do_lower_case

# 将初始化参数 `do_lower_case` 的值保存到实例变量 `self.do_lower_case` 中。


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    """
    Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
    adding special tokens. A Lxmert sequence has the following format:

    - single sequence: `[CLS] X [SEP]`
    - pair of sequences: `[CLS] A [SEP] B [SEP]`

    Args:
        token_ids_0 (`List[int]`):
            List of IDs to which the special tokens will be added.
        token_ids_1 (`List[int]`, *optional*):
            Optional second list of IDs for sequence pairs.

    Returns:
        `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
    """
    output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

    if token_ids_1 is not None:
        output += token_ids_1 + [self.sep_token_id]

    return output

# 构建模型输入的方法，根据单个序列或序列对的情况连接并添加特殊标记。Lxmert 模型序列有特定格式：单个序列包括 `[CLS] X [SEP]`，序列对包括 `[CLS] A [SEP] B [SEP]`。函数接受 `token_ids_0` 和 `token_ids_1` 两个参数，分别是需要添加特殊标记的 ID 列表，返回一个包含输入 ID 及相应特殊标记的列表。


def create_token_type_ids_from_sequences(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None

# 创建生成序列类型标识符 token type IDs 的方法。接受两个参数 `token_ids_0` 和 `token_ids_1`，分别是序列的 ID 列表，返回一个标识序列类型的 ID 列表。
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Lxmert sequence
        pair mask has the following format:

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
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If token_ids_1 is not provided, return a mask with zeros only for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Concatenate the lengths of token_ids_0, sep, and token_ids_1 with appropriate token type IDs
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the tokenizer's vocabulary to the specified directory with an optional filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```