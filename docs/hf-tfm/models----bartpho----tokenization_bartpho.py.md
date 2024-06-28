# `.\models\bartpho\tokenization_bartpho.py`

```py
# 定义脚本编码为 UTF-8

# 版权声明，使用 Apache 许可证 2.0 版本
# 除非符合许可证的要求，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 导入依赖库和模块
""" Tokenization classes for BARTpho-syllable model."""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库

# 导入通用工具模块和日志记录模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# SentencePiece 分词器使用的特殊标记
SPIECE_UNDERLINE = "▁"

# 词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "monolingual_vocab_file": "dict.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/sentencepiece.bpe.model",
    },
    "monolingual_vocab_file": {
        "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/dict.txt",
    },
}

# 预训练模型的位置编码嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"vinai/bartpho-syllable": 1024}


class BartphoTokenizer(PreTrainedTokenizer):
    """
    自 [`XLMRobertaTokenizer`] 改编。基于 [SentencePiece](https://github.com/google/sentencepiece)。

    此分词器继承自 [`PreTrainedTokenizer`]，包含大多数主要方法。用户应参考超类以获取更多有关这些方法的信息。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            每次转换（字符串、标记和 ID）都使用的 SentencePiece 处理器。
    """

    # 词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES

    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 预训练模型的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        monolingual_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 使用 lstrip=True 和 rstrip=False 来确保添加的遮罩标记行为与普通单词相同，即保留其前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置词汇文件和单语词汇文件路径
        self.vocab_file = vocab_file
        self.monolingual_vocab_file = monolingual_vocab_file

        # 使用给定的 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从给定的 vocab_file 中加载 SentencePiece 模型
        self.sp_model.Load(str(vocab_file))

        # 加载减少后的词汇表

        # 保持特殊标记的顺序，以保证向后兼容性
        self.fairseq_tokens_to_ids = {}
        cnt = 0
        # 遍历特殊标记列表，如果标记尚未在 fairseq_tokens_to_ids 中，则将其添加
        for token in [bos_token, pad_token, eos_token, unk_token, sep_token, cls_token]:
            if str(token) not in self.fairseq_tokens_to_ids:
                self.fairseq_tokens_to_ids[str(token)] = cnt
                cnt += 1

        # 从 monolingual_vocab_file 中读取每行的第一个词作为标记，并将其添加到 fairseq_tokens_to_ids 中
        with open(monolingual_vocab_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                token = line.strip().split()[0]
                self.fairseq_tokens_to_ids[token] = len(self.fairseq_tokens_to_ids)

        # 如果 mask_token 尚未在 fairseq_tokens_to_ids 中，则将其添加
        if str(mask_token) not in self.fairseq_tokens_to_ids:
            self.fairseq_tokens_to_ids[str(mask_token)] = len(self.fairseq_tokens_to_ids)

        # 创建 fairseq_ids_to_tokens 字典，用于将 fairseq_tokens_to_ids 的键值对反转
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法，传递必要的参数和关键字参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def __getstate__(self):
        # 复制对象的字典属性
        state = self.__dict__.copy()
        # 将 sp_model 设为 None，以便序列化对象时不包含该属性
        state["sp_model"] = None
        # 获取序列化的 SentencePiece 模型的原型，并存储在 sp_model_proto 中
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        # 将对象的字典属性设置为 d
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 SentencePieceProcessor 对象，并从 sp_model_proto 加载序列化的模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. An BARTPho sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """

        if token_ids_1 is None:
            # Return a single sequence with added special tokens: <s> token_ids_0 </s>
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For sequence pairs, concatenate tokens with special tokens between and at the end: <s> token_ids_0 </s></s> token_ids_1 </s>
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence IDs from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers indicating the presence of special tokens (1) or sequence tokens (0).
        """

        if already_has_special_tokens:
            # If tokens already have special tokens, delegate to superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # For a single sequence, mark positions of special tokens: <s> token_ids_0 </s>
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # For sequence pairs, mark positions of special tokens: <s> token_ids_0 </s></s> token_ids_1 </s>
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        Create token type IDs tensor from sequences for sequence classification tasks. This method assigns each token in the input
        sequences a token type ID (0 or 1) depending on whether it belongs to the first or the second sequence.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: A list of token type IDs where each ID corresponds to the respective input token.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BARTPho does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """
        # Define the separator and class tokens based on the model's special token IDs
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros representing the mask for token_ids_0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Otherwise, return a list of zeros representing the mask for the concatenated sequence pairs
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Return the size of the vocabulary based on the number of entries in fairseq_ids_to_tokens
        return len(self.fairseq_ids_to_tokens)

    def get_vocab(self):
        # Construct and return a dictionary mapping token strings to their respective IDs
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # Update the vocabulary with any additional tokens from added_tokens_encoder
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize the input text using the SentencePiece model and return the tokens as a list of strings
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        # Check if the token exists in fairseq_tokens_to_ids; if not, return the unknown token ID
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        else:
            return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        # Return the token corresponding to the index from fairseq_ids_to_tokens
        return self.fairseq_ids_to_tokens[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string."""
        # Concatenate the tokens into a string, replacing SPIECE_UNDERLINE with space and stripping any surrounding whitespace
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出的词汇表文件路径，根据可选的文件名前缀和预定义的文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 构建输出的单语词汇表文件路径，根据可选的文件名前缀和预定义的文件名
        out_monolingual_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["monolingual_vocab_file"],
        )

        # 如果当前词汇表文件不是输出文件且存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的 sp_model 内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 如果当前单语词汇表文件不是输出文件且存在，则复制当前单语词汇表文件到输出路径
        if os.path.abspath(self.monolingual_vocab_file) != os.path.abspath(
            out_monolingual_vocab_file
        ) and os.path.isfile(self.monolingual_vocab_file):
            copyfile(self.monolingual_vocab_file, out_monolingual_vocab_file)
        # 如果当前单语词汇表文件不存在，则将 fairseq_tokens_to_ids 中的 token 写入输出文件
        elif not os.path.isfile(self.monolingual_vocab_file):
            with open(out_monolingual_vocab_file, "w", encoding="utf-8") as fp:
                for token in self.fairseq_tokens_to_ids:
                    if token not in self.all_special_tokens:
                        fp.write(f"{str(token)} \n")

        # 返回保存的词汇表文件路径和单语词汇表文件路径
        return out_vocab_file, out_monolingual_vocab_file
```