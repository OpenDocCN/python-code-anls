# `.\transformers\models\mbart\tokenization_mbart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI Research Team 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库
import sentencepiece as spm

# 导入所需的自定义库
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义特殊字符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

# Fairseq 语言代码列表
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]  # fmt: skip

# MBartTokenizer 类，继承自 PreTrainedTokenizer
class MBartTokenizer(PreTrainedTokenizer):
    """
    构建一个 MBART 分词器。

    从 `RobertaTokenizer` 和 `XLNetTokenizer` 进行了调整。基于
    [SentencePiece](https://github.com/google/sentencepiece)。

    分词方法为对于源语言文档，为 `<tokens> <eos> <language code>`，对于目标语言文档，为 `<language code>
    <tokens> <eos>`。

    示例：

    ```python
    >>> from transformers import MBartTokenizer

    >>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
    ```py"""

    # 定义词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]

    # 前缀标记列表
    prefix_tokens: List[int] = []
    # 后缀标记列表
    suffix_tokens: List[int] = []
    # 初始化函数，用于创建一个新的对象
    def __init__(
        # 初始化函数的参数列表
        self,  # 指向当前对象的引用
        vocab_file,  # 词汇表文件路径
        bos_token="<s>",  # 开始标记，默认为"<s>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        sep_token="</s>",  # 分隔标记，默认为"</s>"
        cls_token="<s>",  # 类别标记，默认为"<s>"
        unk_token="<unk>",  # 未知标记，默认为"<unk>"
        pad_token="<pad>",  # 填充标记，默认为"<pad>"
        mask_token="<mask>",  # 掩码标记，默认为"<mask>"
        tokenizer_file=None,  # 分词器文件路径，默认为None
        src_lang=None,  # 源语言，默认为None
        tgt_lang=None,  # 目标语言，默认为None
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # sp_model_kwargs参数，默认为None
        additional_special_tokens=None,  # 额外特殊标记，默认为None
        **kwargs,  # 其他关键字参数
        # 如果 mask_token 是字符串类型，则创建一个 AddedToken 对象，保留前面的空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=False) if isinstance(mask_token, str) else mask_token
        )

        # 如果 sp_model_kwargs 为 None，则设为空字典，否则使用传入的参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 使用 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # fairseq 和 spm 词汇必须对齐，建立 fairseq token 到 id 的映射
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = 1

        # 建立语言代码到 id 的映射
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        # 更新 fairseq token 到 id 的映射
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        _additional_special_tokens = list(self.lang_code_to_id.keys())

        # 如果有额外的特殊 token，则添加到 _additional_special_tokens 中
        if additional_special_tokens is not None:
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        # 调用父类的初始化方法
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenizer_file=None,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        # 设置当前源语言和对应的语言代码 id
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)
    # 定义对象的序列化方法，将对象状态保存为字典
    def __getstate__(self):
        # 复制对象的属性字典
        state = self.__dict__.copy()
        # 将 sp_model 属性置为 None
        state["sp_model"] = None
        # 将 sp_model_proto 属性置为序列化的模型协议
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        # 返回对象状态字典
        return state

    # 定义对象的反序列化方法，根据字典恢复对象状态
    def __setstate__(self, d):
        # 将对象的属性字典替换为传入的字典
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 sp_model 对象，并从序列化的模型协议中加载模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 定义属性 vocab_size，返回词汇表大小
    @property
    def vocab_size(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # 加 1 是为了 mask token

    # 定义属性 src_lang，返回源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 定义 src_lang 的 setter 方法，设置源语言并更新特殊标记
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 定义方法 get_special_tokens_mask，获取特殊标记掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        # 如果已经有特殊标记，则调用父类方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 创建前缀和后缀的特殊标记掩码
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        # 如果没有第二个 token_ids，则返回前缀 + token_ids_0 + 后缀
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # 如果有第二个 token_ids，则返回前缀 + token_ids_0 + token_ids_1 + 后缀
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # 定义方法 build_inputs_with_special_tokens，构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 MBART 序列的格式如下，其中 `X` 表示序列：

        - `input_ids`（用于编码器）`X [eos, src_lang_code]`
        - `decoder_input_ids`:（用于解码器）`X [eos, tgt_lang_code]`

        BOS 从未被使用。序列对不是预期的使用情况，但它们将被处理而不使用分隔符。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 具有适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 我们不希望处理序列对，但为了 API 一致性保留了对序列对的逻辑
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。mBART 不使用 token type ids，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """由翻译管道使用，为生成函数准备输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("翻译需要为此模型提供 `src_lang` 和 `tgt_lang`")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)
    def _convert_token_to_id(self, token):
        """将一个标记（字符串）转换为一个 ID，使用词汇表"""
        # 如果标记在 fairseq_tokens_to_ids 中，则返回对应的 ID
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # 否则，使用 sp_model 将标记转换为 ID
        spm_id = self.sp_model.PieceToId(token)

        # 如果 SP 模型返回 0，则返回未知标记的 ID
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """将一个索引（整数）转换为一个标记（字符串），使用词汇表"""
        # 如果索引在 fairseq_ids_to_tokens 中，则返回对应的标记
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # 否则，使用 sp_model 将索引转换为标记
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """将一系列标记（子词的字符串）转换为单个字符串"""
        # 将所有标记连接起来，替换 SPIECE_UNDERLINE 为空格，去除首尾空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """保存词汇表到指定目录"""
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件路径不同且存在，则复制词汇表文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，则将 sp_model 的序列化模型写入文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        """准备用于序列到序列模型的批次数据"""
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        """切换到输入模式"""
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        """切换到目标模式"""
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """设置源语言的特殊标记"""
        # 将当前语言代码设置为源语言的 ID
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        # 设置后缀标记为 [eos, src_lang_code]
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
    # 设置目标语言的特殊标记，重置为目标语言设置。没有前缀，后缀为[eos, tgt_lang_code]。
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        # 将当前语言代码设置为输入语言对应的 ID
        self.cur_lang_code = self.lang_code_to_id[lang]
        # 重置前缀标记为空列表
        self.prefix_tokens = []
        # 设置后缀标记为[eos, tgt_lang_code]
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
```