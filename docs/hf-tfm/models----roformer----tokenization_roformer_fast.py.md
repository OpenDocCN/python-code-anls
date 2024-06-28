# `.\models\roformer\tokenization_roformer_fast.py`

```py
# 导入必要的模块和库
import json  # 导入 json 模块，用于处理 JSON 格式数据
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

from tokenizers import normalizers  # 导入 tokenizers 库中的 normalizers 模块
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer  # 导入 tokenizers 库中的预分词器类

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从上级目录导入 PreTrainedTokenizerFast 类
from ...utils import logging  # 从上级目录导入 logging 模块
from .tokenization_roformer import RoFormerTokenizer  # 从当前目录导入 RoFormerTokenizer 类
from .tokenization_utils import JiebaPreTokenizer  # 从当前目录导入 JiebaPreTokenizer 类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义用于 RoFormer 的词汇文件和 tokenizer 文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射，以及它们对应的下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_char_small": (
            "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_chinese_char_base": (
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_discriminator": (
            "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_generator": (
            "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt"
        ),
    }
}

# 定义预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "junnyu/roformer_chinese_small": 1536,
    "junnyu/roformer_chinese_base": 1536,
    "junnyu/roformer_chinese_char_small": 512,
    "junnyu/roformer_chinese_char_base": 512,
    "junnyu/roformer_small_discriminator": 128,
    "junnyu/roformer_small_generator": 128,
}

# 定义预训练模型的初始化配置映射，指定是否小写化
PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_base": {"do_lower_case": True},
    "junnyu/roformer_small_discriminator": {"do_lower_case": True},
    "junnyu/roformer_small_generator": {"do_lower_case": True},
}


class RoFormerTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" RoFormer tokenizer (backed by HuggingFace's *tokenizers* library).
    # `RoFormerTokenizerFast`几乎与`BertTokenizerFast`相同，实现端到端的分词：
    # 标点符号分割和WordPiece。它们在处理中文时有些差异。
    
    # 此分词器继承自`PreTrainedTokenizerFast`，其中包含大部分主要方法。用户应该
    # 参考这个超类以获取有关这些方法的更多信息。
    
    # 示例：
    #
    # ```
    # >>> from transformers import RoFormerTokenizerFast
    #
    # >>> tokenizer = RoFormerTokenizerFast.from_pretrained("junnyu/roformer_chinese_base")
    # >>> tokenizer.tokenize("今天天气非常好。")
    # ['今', '天', '天', '气', '非常', '好', '。']
    # ```
    
    vocab_files_names = VOCAB_FILES_NAMES  # 获取词汇文件的名称列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 获取预训练词汇文件的映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 获取预训练位置嵌入的最大模型输入尺寸
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 获取预训练初始化配置
    slow_tokenizer_class = RoFormerTokenizer  # 慢速分词器类为RoFormerTokenizer
    
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
        # 调用父类的初始化方法，设置基本的分词器参数
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
    
        # 从后端分词器的normalizer状态中加载JSON数据
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 如果normalizer的lowercase属性与当前设置不符，则更新
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            # 更新后端分词器的normalizer
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)
    
        # 确保正确设置自定义的PreTokenizer
        vocab = self.backend_tokenizer.get_vocab()
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))
    
        self.do_lower_case = do_lower_case
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将分词器的pre_tokenizer设置为BertPreTokenizer()
        state["_tokenizer"].pre_tokenizer = BertPreTokenizer()
        return state
    
    def __setstate__(self, d):
        self.__dict__ = d
        # 获取当前分词器的词汇表
        vocab = self.__dict__["_tokenizer"].get_vocab()
        # 将分词器的pre_tokenizer设置为自定义的JiebaPreTokenizer
        self.__dict__["_tokenizer"].pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer(vocab))
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

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
        # Initialize output with CLS token ID, token_ids_0, and SEP token ID
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # If token_ids_1 is provided, concatenate token_ids_1 and SEP token ID
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
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
        # Define SEP and CLS tokens as lists
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros corresponding to token_ids_0 + CLS + SEP
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return a concatenated list of zeros for token_ids_0 + CLS + SEP and ones for token_ids_1 + SEP
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to a directory.

        Args:
            save_directory (str):
                Directory to save the vocabulary files.
            filename_prefix (str, *optional*):
                Prefix for the vocabulary files.

        Returns:
            `Tuple[str]`: Tuple of file paths where the vocabulary was saved.
        """
        # Save the model vocabulary using the tokenizer's save method
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        """
        Save the pretrained model and its tokenizer.

        Args:
            save_directory (str):
                Directory to save the pretrained model.
            legacy_format (str, *optional*):
                Legacy format compatibility.
            filename_prefix (str, *optional*):
                Prefix for the saved files.
            push_to_hub (bool):
                Whether to push the saved model to the Hugging Face model hub.
            **kwargs:
                Additional arguments passed to the superclass method.

        Returns:
            `Any`: Output of the superclass's `save_pretrained` method.
        """
        # Set the pre_tokenizer to BertPreTokenizer before saving
        self.backend_tokenizer.pre_tokenizer = BertPreTokenizer()
        
        # Call the superclass's save_pretrained method with the specified arguments
        return super().save_pretrained(save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
```