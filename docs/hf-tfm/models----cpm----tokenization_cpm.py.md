# `.\models\cpm\tokenization_cpm.py`

```py
# coding=utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，许可证地址为 http://www.apache.org/licenses/LICENSE-2.0
# Tokenization 类

import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
# 导入 sentencepiece 和 logging 模块
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import SPIECE_UNDERLINE, logging

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 设置词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}
# 设置预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    }
}

class CpmTokenizer(PreTrainedTokenizer):
    """使用结巴分词工具进行预分词处理。用于 CPM 模型。"""

    # 词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 初始化函数
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    # 获取词汇表大小
    @property
    def vocab_size(self):
        return len(self.sp_model)

    # 获取词汇表
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 获取状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 设置状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 后向兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
    # 从XLNetTokenizer.preprocess_text中复制过来，预处理输入文本数据
    def preprocess_text(self, inputs):
        如果配置了去除空格选项，则去除文本中多余的空格
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        否则直接使用原始文本数据
        else:
            outputs = inputs
        将文本中的"``"和"''"替换为双引号
        outputs = outputs.replace("``", '"').replace("''", '"')

        如果不保留重音符号，则进行 Unicode 规范化处理和过滤
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        如果要求小写处理，则将文本全部转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        返回处理后的文本数据
        return outputs

    # 从XLNetTokenizer._tokenize中复制过来，对文本进行分词处理
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        对输入文本进行预处理
        text = self.preprocess_text(text)
        使用 sp_model 对文本进行编码，获取词片段
        pieces = self.sp_model.encode(text, out_type=str)
        初始化一个新的词片段列表
        new_pieces = []
        遍历原词片段列表进行处理
        for piece in pieces:
            如果词片段长度大于1且结尾是逗号和数字，则对词片段进行特殊处理
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                将符合条件的词片段拆分并添加到新的词片段列表中
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                如果第一个词片段不是引号_ 开头并且拆分后第一个词片段是引号_ 开头，则修正处理
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    如果第一个词片段长度为1，则去掉第一个词片段
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    否则去掉引号_ 开头再添加到新的词片段列表中
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                将原词片段的逗号添加到新的词片段列表中
                cur_pieces.append(piece[-1])
                扩展新的词片段列表
                new_pieces.extend(cur_pieces)
            否则直接添加原词片段到新的词片段列表中
            else:
                new_pieces.append(piece)

        返回处理后的新的词片段列表
        return new_pieces

    # 从XLNetTokenizer._convert_token_to_id中复制过来，将 token 转换为 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        使用 sp_model 将 token 转换为 id
        return self.sp_model.PieceToId(token)

    # 从XLNetTokenizer._convert_id_to_token中复制过来，将 id 转换为 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        使用 sp_model 将 index 转换为 token
        return self.sp_model.IdToPiece(index)

    # 从XLNetTokenizer.convert_tokens_to_string中复制过来，将 token 序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        将 token 序列连接为字符串并替换掉 SPIECE_UNDERLINE，然后去掉首尾空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        返回连接后的字符串
        return out_string

    # 从XLNetTokenizer.build_inputs_with_special_tokens中复制过来，构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，为序列分类任务构建模型输入。XLNet 序列的格式如下：

        - 单个序列：`X <sep> <cls>`
        - 序列对：`A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表（可选）。

        Returns:
            `List[int]`: 带有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        sep = [self.sep_token_id]  # 获取分隔符标记的 ID
        cls = [self.cls_token_id]  # 获取类标记的 ID
        if token_ids_1 is None:
            return token_ids_0 + sep + cls  # 返回单个序列的输入 ID 列表
        return token_ids_0 + sep + token_ids_1 + sep + cls  # 返回序列对的输入 ID 列表

    # 从 transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.get_special_tokens_mask 复制过来
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表（可选）。
            already_has_special_tokens (`bool`, *可选*，默认为 `False`):
                标记列表是否已经使用特殊标记格式化为模型。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        return ([0] * len(token_ids_0)) + [1, 1]

    # 从 transformers.models.xlnet.tokenization_xlnet.XLNetTokenizer.create_token_type_ids_from_sequences 复制过来
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
```  
    # 定义函数，创建用于序列对分类任务的 mask
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:

        ```py
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
        # 定义分隔符和指示第二个序列的 ID
        sep = [self.sep_token_id]
        cls_segment_id = [2]

        # 如果 token_ids_1 为空，则返回第一个序列的 mask
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        # 返回两个序列的 mask
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
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
        # 如果词汇表文件不存在，则从 sp_model 中写入词汇表内容
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # 解码函数，去除空格和特殊符号
    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        return text
```