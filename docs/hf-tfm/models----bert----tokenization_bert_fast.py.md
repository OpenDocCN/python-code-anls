# `.\transformers\models\bert\tokenization_bert_fast.py`

```
# 设定文件编码为 UTF-8

# 版权声明及许可证信息，表示此代码的版权和许可

# 导入必要的模块
import json  # 导入 JSON 模块，用于处理 JSON 数据
from typing import List, Optional, Tuple  # 导入 typing 模块中的 List、Optional、Tuple 类型

# 导入 tokenizers 模块中的 normalizers 子模块
from tokenizers import normalizers  

# 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  

# 导入 utils 模块中的 logging 子模块
from ...utils import logging  

# 从当前目录的 tokenization_bert 模块中导入 BertTokenizer 类
from .tokenization_bert import BertTokenizer  


# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
```  
    # 词汇表文件链接字典，包含各种不同版本的BERT模型和其对应的词汇表文件链接
    "vocab_file": {
        # bert-base-uncased 模型的词汇表文件链接
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        # bert-large-uncased 模型的词汇表文件链接
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        # bert-base-cased 模型的词汇表文件链接
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        # bert-large-cased 模型的词汇表文件链接
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        # bert-base-multilingual-uncased 模型的词汇表文件链接
        "bert-base-multilingual-uncased": (
            "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        # bert-base-multilingual-cased 模型的词汇表文件链接
        "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        # bert-base-chinese 模型的词汇表文件链接
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        # bert-base-german-cased 模型的词汇表文件链接
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
        # bert-large-uncased-whole-word-masking 模型的词汇表文件链接
        "bert-large-uncased-whole-word-masking": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
        ),
        # bert-large-cased-whole-word-masking 模型的词汇表文件链接
        "bert-large-cased-whole-word-masking": (
            "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
        ),
        # bert-large-uncased-whole-word-masking-finetuned-squad 模型的词汇表文件链接
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        # bert-large-cased-whole-word-masking-finetuned-squad 模型的词汇表文件链接
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        # bert-base-cased-finetuned-mrpc 模型的词汇表文件链接
        "bert-base-cased-finetuned-mrpc": (
            "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
        ),
        # bert-base-german-dbmdz-cased 模型的词汇表文件链接
        "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        # bert-base-german-dbmdz-uncased 模型的词汇表文件链接
        "bert-base-german-dbmdz-uncased": (
            "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
        ),
        # TurkuNLP/bert-base-finnish-cased-v1 模型的词汇表文件链接
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
        ),
        # TurkuNLP/bert-base-finnish-uncased-v1 模型的词汇表文件链接
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
        ),
        # wietsedv/bert-base-dutch-cased 模型的词汇表文件链接
        "wietsedv/bert-base-dutch-cased": (
            "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
        ),
    },
    # 定义一个字典，用于存储不同 BERT 模型的 tokenizer 文件的下载链接
    "tokenizer_file": {
        # BERT 模型 "bert-base-uncased" 对应的 tokenizer 文件下载链接
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/tokenizer.json",
        # BERT 模型 "bert-large-uncased" 对应的 tokenizer 文件下载链接
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/tokenizer.json",
        # BERT 模型 "bert-base-cased" 对应的 tokenizer 文件下载链接
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json",
        # BERT 模型 "bert-large-cased" 对应的 tokenizer 文件下载链接
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/tokenizer.json",
        # BERT 模型 "bert-base-multilingual-uncased" 对应的 tokenizer 文件下载链接
        "bert-base-multilingual-uncased": (
            "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-base-multilingual-cased" 对应的 tokenizer 文件下载链接
        "bert-base-multilingual-cased": (
            "https://huggingface.co/bert-base-multilingual-cased/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-base-chinese" 对应的 tokenizer 文件下载链接
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/tokenizer.json",
        # BERT 模型 "bert-base-german-cased" 对应的 tokenizer 文件下载链接
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/tokenizer.json",
        # BERT 模型 "bert-large-uncased-whole-word-masking" 对应的 tokenizer 文件下载链接
        "bert-large-uncased-whole-word-masking": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-large-cased-whole-word-masking" 对应的 tokenizer 文件下载链接
        "bert-large-cased-whole-word-masking": (
            "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-large-uncased-whole-word-masking-finetuned-squad" 对应的 tokenizer 文件下载链接
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-large-cased-whole-word-masking-finetuned-squad" 对应的 tokenizer 文件下载链接
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-base-cased-finetuned-mrpc" 对应的 tokenizer 文件下载链接
        "bert-base-cased-finetuned-mrpc": (
            "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-base-german-dbmdz-cased" 对应的 tokenizer 文件下载链接
        "bert-base-german-dbmdz-cased": (
            "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "bert-base-german-dbmdz-uncased" 对应的 tokenizer 文件下载链接
        "bert-base-german-dbmdz-uncased": (
            "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "TurkuNLP/bert-base-finnish-cased-v1" 对应的 tokenizer 文件下载链接
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "TurkuNLP/bert-base-finnish-uncased-v1" 对应的 tokenizer 文件下载链接
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/tokenizer.json"
        ),
        # BERT 模型 "wietsedv/bert-base-dutch-cased" 对应的 tokenizer 文件下载链接
        "wietsedv/bert-base-dutch-cased": (
            "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型的位置嵌入大小字典，键为模型名称，值为位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

# 预训练模型的初始化配置字典，键为模型名称，值为初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}

# 定义 BertTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class BertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" BERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在标记化时是否将输入转换为小写。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记无法转换为 ID，而是设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，在从多个序列构建序列时使用，例如用于序列分类的两个序列或用于文本和问题的问题回答。在使用特殊标记构建的序列的最后一个标记也会使用此标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，在进行序列分类（整个序列的分类而不是每个标记的分类）时使用。在使用特殊标记构建的序列中是第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于屏蔽值的标记。在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        clean_text (`bool`, *optional*, defaults to `True`):
            在标记化之前是否清理文本，通过删除所有控制字符并将所有空格替换为经典空格。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。对于日语，这可能应该被禁用（参见[此问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由`lowercase`的值确定（与原始BERT相同）。
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            子词的前缀。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = BertTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化模型的tokenizer
        super().__init__(
            vocab_file,  # 词汇表文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            do_lower_case=do_lower_case,  # 是否将输入文本转换为小写
            unk_token=unk_token,  # 未知标记
            sep_token=sep_token,  # 分隔标记
            pad_token=pad_token,  # 填充标记
            cls_token=cls_token,  # 类别标记
            mask_token=mask_token,  # 掩码标记
            tokenize_chinese_chars=tokenize_chinese_chars,  # 是否对中文字符进行分词
            strip_accents=strip_accents,  # 是否去除重音符号
            **kwargs,  # 其他参数
        )

        # 将 backend_tokenizer 的正则化器状态转换为 JSON 格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查是否需要更新正则化器的配置
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取正则化器类
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新正则化器状态
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 使用新的配置重新初始化正则化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新实例属性 do_lower_case
        self.do_lower_case = do_lower_case

    # 为序列分类任务构建模型输入，包括特殊标记
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

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
        # 构建带有特殊标记的输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从序列构建令牌类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
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
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator token ID
        sep = [self.sep_token_id]
        # Define the classification token ID
        cls = [self.cls_token_id]
        # If token_ids_1 is None, return a mask with only the first portion (0s)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Return a mask with both sequences, using 0s for the first sequence and 1s for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the vocabulary files to the specified directory with the given filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the list of saved files as a tuple
        return tuple(files)
```