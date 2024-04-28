# `.\transformers\models\camembert\tokenization_camembert.py`

```py
# 设置文件编码为 utf-8
# 版权声明，包括作者和许可证信息
# 该代码基于 Apache License, Version 2.0 许可证
# 详细许可证信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 除非符合许可证要求，否则不得使用此文件
# 该代码是基于 Google AI、Google Brain、Carnegie Mellon University 作者和 HuggingFace Inc. 团队的工作
# 未经许可，不得使用该文件
# 该代码用于 Camembert 模型的分词类
# 导入所需的库和模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
# 获取日志记录器
logger = logging.get_logger(__name__)
# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}
# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "camembert-base": "https://huggingface.co/camembert-base/resolve/main/sentencepiece.bpe.model",
    }
}
# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "camembert-base": 512,
}
# SentencePiece 分词中的特殊标记
SPIECE_UNDERLINE = "▁"

# CamembertTokenizer 类，继承自 PreTrainedTokenizer
class CamembertTokenizer(PreTrainedTokenizer):
    """
    从 RobertaTokenizer 和 XLNetTokenizer 改编而来。构建一个 CamemBERT 分词器。基于 SentencePiece。

    该分词器继承自 PreTrainedTokenizer，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """
    # 定义词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 定义一个方法，接受一个参数并不返回任何内容
        # Mask token行为类似于普通单词，即包括它前面的空格
        mask_token = (
            # 如果mask_token是字符串，则创建一个AddedToken对象，剥离左边的空格，保留右边的空格，不进行标准化，表示为特殊字符
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False, special=True)
            # 如果mask_token不是字符串，则直接使用传入的mask_token对象
            if isinstance(mask_token, str)
            else mask_token
        )

        # 如果sp_model_kwargs为None，则设置为一个空字典，否则保持原样
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 使用给定的sp_model_kwargs参数创建SentencePieceProcessor对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载指定的词汇文件
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # HACK: 这些令牌是作者为了某种不明原因而添加的，因为它们已经是sentencepiece词汇的一部分（对于<s>和</s>以及<unk>是这样）。
        # 在这种情况下，建议手动设置令牌。
        # 将已添加的令牌映射到其索引
        self._added_tokens_decoder = {
            0: AddedToken("<s>NOTUSED", special=True),
            # 如果pad_token是字符串，则创建一个AddedToken对象，表示为特殊字符
            1: AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token,
            2: AddedToken("</s>NOTUSED", special=True),
            # 如果unk_token是字符串，则创建一个AddedToken对象，表示为特殊字符
            3: AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token,
            4: AddedToken("<unk>NOTUSED", special=True),
        }

        # 设置fairseq_offset为4，表示有3个新添加的令牌，但偏移量从4开始
        self.fairseq_offset = 4

        # legacy: camemebert是一个特殊情况，我们必须确保`"<unk>NOTUSED"`在这里
        if "added_tokens_decoder" in kwargs:
            # 这是唯一需要这样做的类......
            # 原因是快速版本有一个完整的。
            # 更新已添加的令牌映射
            kwargs["added_tokens_decoder"].update(self._added_tokens_decoder)

        # 调用父类的构造函数，传入相应的参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇的大小，包括已添加的令牌
        return len(self.sp_model)

    def get_vocab(self):
        # 获取词汇表，包括已添加的令牌
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.fairseq_offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # 使用SentencePiece对文本进行分词
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果token在词汇中不存在，则返回unk_token的id
        if self.sp_model.PieceToId(token) == 0:
            # 将sentence piece unk token转换为fairseq unk token的索引
            return self.unk_token_id
        # 返回token在词汇中的索引加上fairseq_offset
        return self.fairseq_offset + self.sp_model.PieceToId(token)
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将索引（整数）转换为标记（字符串）
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列标记（字符串）转换为单个字符串
        # TODO decode outputs do not match between fast and slow
        # 用于将标记序列合并成单个字符串，存在快慢解码不匹配的问题
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊标记不使用sentencepiece模型进行解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        # 用于向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用sp_model_kwargs参数创建SentencePieceProcessor对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到sp_model
        self.sp_model.Load(self.vocab_file)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则返回错误信息
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出的词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇文件的绝对路径与输出文件路径不同且词汇文件存在，则复制词汇文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇文件不存在，则将serialized_model_proto写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An CamemBERT sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # Check if only one sequence is provided
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # Create special token lists for the beginning, end, and separator
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # Return the concatenated list of tokens with special tokens added
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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
        # Check if special tokens are already present in the token lists
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If only one sequence is provided, add special tokens at the beginning and end
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # If two sequences are provided, add special tokens at the beginning and end of each sequence
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
        RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 创建一个用于序列对分类任务的掩码。CamemBERT与RoBERTa一样，不使用token type ids，因此返回一个零列表。
        sep = [self.sep_token_id]  # 获取分隔符对应的token id
        cls = [self.cls_token_id]  # 获取起始符对应的token id

        if token_ids_1 is None:  # 如果只有一个序列
            return len(cls + token_ids_0 + sep) * [0]  # 返回零列表，长度为起始符、第一个序列和分隔符的长度之和
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 如果有两个序列，返回零列表，长度为起始符、第一个序列、两个分隔符和第二个序列的长度之和
```