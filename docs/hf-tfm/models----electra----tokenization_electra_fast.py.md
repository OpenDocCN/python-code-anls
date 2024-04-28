# `.\models\electra\tokenization_electra_fast.py`

```
# 导入json模块，用于处理JSON数据
import json
# 从typing模块中导入List、Optional和Tuple，用于类型提示
from typing import List, Optional, Tuple
# 从tokenizers模块中导入normalizers，用于规范化文本
from tokenizers import normalizers
# 从tokenization_utils_fast模块中导入PreTrainedTokenizerFast类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从当前目录下的tokenization_electra模块中导入ElectraTokenizer类
from .tokenization_electra import ElectraTokenizer

# 定义存储词汇表文件名的字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇表文件的映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # 生成器模型的词汇表文件URL
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
        # 生成器模型的分词器文件URL
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

# 定义预训练位置嵌入大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    # 定义了一组键值对，键是模型名称，值是模型的大小（512）
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
# 结束代码块
}

# 预训练模型的初始化配置，包含了各种不同大小的生成器和鉴别器模型
PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制的代码，将Bert相关的部分替换为Electra
class ElectraTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”ELECTRA分词器（由HuggingFace的*tokenizers*库支持）。基于WordPiece。

    这个分词器继承自[`PreTrainedTokenizerFast`]，其中包含了大部分主要方法。用户应该参考这个超类获取更多关于这些方法的信息。
    用于初始化 ElectraTokenizer 类的构造函数，用于创建 ElectraTokenizer 对象
    Args:
        vocab_file (`str`): 包含词汇的文件路径
        do_lower_case (`bool`, *optional*, 默认为 `True`): 在进行标记化时是否将输入转换为小写。
        unk_token (`str`, *optional*, 默认为 `"[UNK]"`): 未知标记。词汇表中不存在的标记无法转换为 ID，将被设置为此标记。
        sep_token (`str`, *optional*, 默认为 `"[SEP]"`): 分隔符标记，用于从多个序列构建序列时使用，例如序列分类或问答中的文本和问题。在使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, 默认为 `"[PAD]"`): 用于填充的标记，在对不同长度的序列进行批处理时使用。
        cls_token (`str`, *optional*, 默认为 `"[CLS]"`): 分类器标记，用于进行序列分类（整个序列的分类，而不是每个标记的分类）。在使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, 默认为 `"[MASK]"`): 用于屏蔽值的标记。这是在训练具有遮罩语言建模的模型时使用的标记。模型将尝试预测此标记。
        clean_text (`bool`, *optional*, 默认为 `True`): 在进行标记化之前是否清理文本，删除任何控制字符并将所有空格替换为经典空格。
        tokenize_chinese_chars (`bool`, *optional*, 默认为 `True`): 是否标记化中文字符。这可能应该在日本语中禁用（请参见 [此问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*): 是否去除所有音调符号。如果未指定此选项，则将通过 `lowercase` 的值（如原始 ELECTRA 中）来确定。
        wordpieces_prefix (`str`, *optional*, 默认为 `##`): 子词的前缀。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = ElectraTokenizer

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
        # 调用父类的构造函数，传入参数初始化对象
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

        # 将后端分词器的规范化器的状态转换为 JSON 格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查规范化器状态是否与传入的参数一致，如果不一致则修改
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

        # 设置对象属性值
        self.do_lower_case = do_lower_case

    # 构造包含特殊标记的输入序列
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
        # 构建包含特殊标记的输入序列
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有第二个序列的 token_ids，则加入对应的特殊标记
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从输入序列创建 token 类型 ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ELECTRA sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
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
        # 创建一个序列对掩码，用于序列对分类任务
        sep = [self.sep_token_id] # 设置分隔符标记的列表
        cls = [self.cls_token_id] # 设置类别标记的列表
        if token_ids_1 is None: # 如果第二个序列的标记列表为空
            # 返回第一个序列的掩码，即长度为第一个序列长度加上分隔符长度加上类别标记长度的0列表
            return len(cls + token_ids_0 + sep) * [0]
        # 如果第二个序列的标记列表不为空
        # 返回第一个序列和第二个序列的掩码，即前面的掩码加上第二个序列的长度加上分隔符的长度的1列表
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用tokenizer的model.save方法来保存词汇表到指定目录，使用指定的文件名前缀
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名组成的元组
        return tuple(files)
```