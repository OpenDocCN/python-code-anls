# `.\transformers\models\albert\tokenization_albert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google AI、Google Brain 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" ALBERT 模型的分词类 """

# 导入所需的库
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 导入所需的模块和函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/spiece.model",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/spiece.model",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/spiece.model",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/spiece.model",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/spiece.model",
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert-base-v1": 512,
    "albert-large-v1": 512,
    "albert-xlarge-v1": 512,
    "albert-xxlarge-v1": 512,
    "albert-base-v2": 512,
    "albert-large-v2": 512,
    "albert-xlarge-v2": 512,
    "albert-xxlarge-v2": 512,
}

# SentencePiece 分词符号
SPIECE_UNDERLINE = "▁"

# ALBERT 分词器类，继承自 PreTrainedTokenizer
class AlbertTokenizer(PreTrainedTokenizer):
    """
    构建一个 ALBERT 分词器。基于 SentencePiece。

    该分词器继承自 PreTrainedTokenizer，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    # 定义词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化函数，设置各种参数和属性
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果 mask_token 是字符串类型，则创建一个 AddedToken 对象
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 如果 sp_model_kwargs 为 None，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置各种属性
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 spm 库创建 SentencePieceProcessor 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化函数
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # 返回词汇表大小
    @property
    def vocab_size(self) -> int:
        return len(self.sp_model)

    # 获取词汇表
    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 序列化对象时调用，返回对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 反序列化对象时调用，设置对象的状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 spm 库创建 SentencePieceProcessor 对��，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 对输入文本进行预处理
    def preprocess_text(self, inputs):
        # 如果需要去除空格，则去除空格并重新连接
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        # 替换特殊字符
        outputs = outputs.replace("``", '"').replace("''", '"')

        # 如果不保留重音符号，则进行 Unicode 规范化和去除重音符号
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        # 如果需要转换为小写，则转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs
    # 将字符串进行分词处理
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 预处理文本
        text = self.preprocess_text(text)
        # 使用句子分割模型对文本进行编码
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            # 处理特殊情况的逻辑，参考链接 https://github.com/google-research/bert/blob/master/README.md#tokenization
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 对特殊情况进行处理，例如 `9,9` -> ['▁9', ',', '9'] 而不是 [`_9,`, '9']
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    # 将 token 转换为 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    # 将 id 转换为 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊 token 不使用 sentencepiece 模型解码
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

    # 构建带有特殊 token 的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。ALBERT 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 一对序列：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
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
        # 定义分隔符的 ID 列表
        sep = [self.sep_token_id]
        # 定义类别标识符的 ID 列表
        cls = [self.cls_token_id]

        # 如果第二个序列的 ID 列表为空
        if token_ids_1 is None:
            # 返回只包含第一个序列和分隔符的部分的标记类型 ID 列表，长度为 cls + token_ids_0 + sep 的长度，全部填充为 0
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回完整的标记类型 ID 列表，前半部分对应第一个序列，后半部分对应第二个序列
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则报错并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 定义输出的词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件路径与输出词汇文件路径不同，并且当前词汇文件存在
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            # 复制当前词汇文件到输出词汇文件路径
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在
        elif not os.path.isfile(self.vocab_file):
            # 将序列化的 sp_model 内容写入输出词汇文件
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出词汇文件路径
        return (out_vocab_file,)
```