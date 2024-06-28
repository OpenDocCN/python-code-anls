# `.\models\electra\tokenization_electra_fast.py`

```
# 导入必要的模块
import json  # 导入用于处理 JSON 数据的模块
from typing import List, Optional, Tuple  # 导入类型提示模块

from tokenizers import normalizers  # 从 tokenizers 模块导入 normalizers 功能

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练分词器
from .tokenization_electra import ElectraTokenizer  # 从当前目录下的 tokenization_electra 模块导入 ElectraTokenizer 类

# 定义文件名与文件路径映射关系的常量字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型与其词汇文件和分词器文件映射关系的常量字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/electra-small-generator": (
            "https://huggingface.co/google/electra-small-generator/resolve/main/vocab.txt"
        ),
        "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
        "google/electra-large-generator": (
            "https://huggingface.co/google/electra-large-generator/resolve/main/vocab.txt"
        ),
        "google/electra-small-discriminator": (
            "https://huggingface.co/google/electra-small-discriminator/resolve/main/vocab.txt"
        ),
        "google/electra-base-discriminator": (
            "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt"
        ),
        "google/electra-large-discriminator": (
            "https://huggingface.co/google/electra-large-discriminator/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "google/electra-small-generator": (
            "https://huggingface.co/google/electra-small-generator/resolve/main/tokenizer.json"
        ),
        "google/electra-base-generator": (
            "https://huggingface.co/google/electra-base-generator/resolve/main/tokenizer.json"
        ),
        "google/electra-large-generator": (
            "https://huggingface.co/google/electra-large-generator/resolve/main/tokenizer.json"
        ),
        "google/electra-small-discriminator": (
            "https://huggingface.co/google/electra-small-discriminator/resolve/main/tokenizer.json"
        ),
        "google/electra-base-discriminator": (
            "https://huggingface.co/google/electra-base-discriminator/resolve/main/tokenizer.json"
        ),
        "google/electra-large-discriminator": (
            "https://huggingface.co/google/electra-large-discriminator/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练模型与其位置嵌入大小的映射关系的常量字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    # 定义一个字典，包含四个条目，每个条目的键是一个字符串表示的模型名称，值是一个整数表示的模型大小（512表示模型大小为512字节）
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}

# 预定义的预训练配置字典，包含了Electra模型的不同预训练变体及其配置信息
PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast复制而来，修改为支持Electra模型的快速分词器
class ElectraTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”的ELECTRA分词器（基于HuggingFace的*tokenizers*库），基于WordPiece。

    此分词器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考该超类获取更多关于这些方法的信息。
    ```
    # 定义一个类，实现ElectraTokenizer的功能
    class ElectraTokenizer:
        # 默认的词汇文件名列表
        vocab_files_names = VOCAB_FILES_NAMES
        # 预训练模型的词汇文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 预训练模型的初始化配置
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        # 预训练位置嵌入的最大模型输入大小
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # ElectraTokenizer 的慢速实现类
        slow_tokenizer_class = ElectraTokenizer
    
        # 初始化方法，用于创建一个 ElectraTokenizer 对象
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

# 调用父类的初始化方法，传入必要的参数和关键字参数来初始化对象。


        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())

# 从后端的分词器对象中获取标准化器的状态，将其反序列化为Python对象。


        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):

# 检查标准化器的状态是否与当前对象的参数匹配，如果不匹配则需要更新标准化器。


            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

# 如果有不匹配的参数，根据标准化器的类型更新标准化器对象，确保与当前对象的参数一致。


        self.do_lower_case = do_lower_case

# 更新当前对象的小写参数为传入的do_lower_case值。


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A ELECTRA sequence has the following format:

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

# 构建模型输入，根据输入的序列或序列对进行连接并添加特殊标记，用于序列分类任务。ELECTRA序列的格式包括单一序列和序列对，对应不同的特殊标记。


        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

# 如果提供了第二个序列token_ids_1，则将其连接到output中并添加特殊分隔标记，最后返回构建好的输入列表。


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None

# 根据给定的序列创建token type IDs，用于区分不同序列的类型。
    def create_electra_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ELECTRA sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define the separation and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is not provided, return a mask with zeros for the first sequence only
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # Return a mask with zeros for the first sequence and ones for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary files to the specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary files will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved files.

        Returns:
            `Tuple[str]`: Tuple containing the filenames of the saved vocabulary files.
        """
        # Save the model's vocabulary files using the tokenizer's internal method
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```