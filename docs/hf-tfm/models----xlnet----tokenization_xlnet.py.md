# `.\models\xlnet\tokenization_xlnet.py`

```py
# coding=utf-8
# 代码文件的版权声明和许可信息，遵循 Apache License, Version 2.0
# 详细信息可查看 http://www.apache.org/licenses/LICENSE-2.0

# 导入标准库和第三方库
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，用于分词
import sentencepiece as spm

# 导入自定义的模块和工具函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import SPIECE_UNDERLINE, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/spiece.model",
        "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/spiece.model",
    }
}

# 定义预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlnet/xlnet-base-cased": None,
    "xlnet/xlnet-large-cased": None,
}

# 定义各个语段的标识符常量
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

# XLNetTokenizer 类，继承自 PreTrainedTokenizer 类
class XLNetTokenizer(PreTrainedTokenizer):
    """
    构建一个 XLNet 分词器，基于 SentencePiece。

    该分词器继承自 `PreTrainedTokenizer`，其中包含大多数主要方法。用户应参考该超类以获取关于这些方法的更多信息。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于所有转换（字符串、token 和 ID）的 SentencePiece 处理器。
    """

    # 类属性：词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES

    # 类属性：预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 类属性：预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 类属性：填充位置为左侧
    padding_side = "left"

    # 初始化方法
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
    ) -> None:
        # 定义一个函数，初始化一个 Mask token 对象，可以作为普通单词处理，即保留前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        # 初始化参数，如果没有传入 sp_model_kwargs，则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置对象属性
        self.do_lower_case = do_lower_case  # 是否进行小写处理
        self.remove_space = remove_space    # 是否移除空格
        self.keep_accents = keep_accents    # 是否保留重音符号
        self.vocab_file = vocab_file        # 词汇文件路径

        # 使用 SentencePieceProcessor 初始化 self.sp_model 对象，传入 sp_model_kwargs 参数
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)  # 载入指定的词汇文件

        # 调用父类的初始化方法，传入多个参数和关键字参数
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
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        # 设置内部变量 _pad_token_type_id 的值为 3
        self._pad_token_type_id = 3

    @property
    def vocab_size(self):
        # 返回 self.sp_model 中的词汇大小
        return len(self.sp_model)

    def get_vocab(self):
        # 创建词汇表字典，将词汇 ID 映射为对应的 token，并更新额外特殊 token 的编码
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 复制对象的状态字典
        state = self.__dict__.copy()
        state["sp_model"] = None  # 将 sp_model 设置为 None，用于对象序列化时的状态保存
        return state

    def __setstate__(self, d):
        # 恢复对象的状态
        self.__dict__ = d

        # 为了向后兼容性，在恢复状态后，如果没有 sp_model_kwargs 属性，则设为一个空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新初始化 self.sp_model 对象，载入指定的词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def preprocess_text(self, inputs):
        # 预处理文本函数，根据对象的属性进行文本处理

        # 如果 remove_space 为 True，则移除多余的空格
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        # 替换特定的引号符号
        outputs = outputs.replace("``", '"').replace("''", '"')

        # 如果不保留重音符号，则使用 NFC 规范化和移除所有的组合字符
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        # 如果 do_lower_case 为 True，则将文本转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 对输入文本进行预处理
        text = self.preprocess_text(text)
        # 使用预训练的分词模型对文本进行分词，返回分词后的结果
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        # 遍历每个分词结果
        for piece in pieces:
            # 如果分词长度大于1且以逗号结尾并且倒数第二个字符是数字
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 对满足条件的分词进行进一步处理，去除特殊字符并拆分为更小的片段
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                # 如果原始分词不以特殊字符开头但当前分词片段以特殊字符开头，则进行调整
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                # 将处理后的分词片段添加到新的分词列表中
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                # 将不需要进一步处理的分词直接添加到新的分词列表中
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用预训练的分词模型将分词转换为对应的 id
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用预训练的分词模型将 id 转换为对应的分词
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 将分词序列转换为单个字符串，并替换特殊字符为空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ):
        """Decode a list of token IDs back into a string."""
        # 略
    ) -> str:
        # 从 kwargs 中弹出 "use_source_tokenizer" 参数，默认为 False
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 将 token_ids 转换为 tokens，跳过特殊 token
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 避免混合字节级和 Unicode 的情况，特别是对于字节级 BPT
        # 需要分别构建添加的 token 和字节级 token 的字符串
        # 参考：https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                # 如果当前有未处理完的子字符串，先转换为字符串并添加到 sub_texts
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                # 直接添加特殊 token 到 sub_texts
                sub_texts.append(token)
            else:
                # 将普通 token 添加到当前的子字符串
                current_sub_text.append(token)
        # 处理最后一个子字符串
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 模仿 Rust 分词器的行为：
        # 默认情况下，特殊 token 之间没有空格
        text = "".join(sub_texts)

        # 是否清理 tokenization 中的空格，默认为 self.clean_up_tokenization_spaces
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        # 如果需要清理空格，则调用 clean_up_tokenization 方法清理 text
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊 token，构建用于序列分类任务的模型输入。对于 XLNet，序列的格式如下：

        - 单个序列：`X <sep> <cls>`
        - 序列对：`A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊 token 的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊 token 的输入 ID 列表。
        """
        sep = [self.sep_token_id]  # 分隔 token 的 ID 列表
        cls = [self.cls_token_id]  # 类别 token 的 ID 列表
        if token_ids_1 is None:
            return token_ids_0 + sep + cls  # 返回单个序列的输入 ID 列表
        return token_ids_0 + sep + token_ids_1 + sep + cls  # 返回序列对的输入 ID 列表

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        生成特殊 token 的掩码，用于标识输入中的特殊 token。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 token ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 token ID 列表。
            already_has_special_tokens (`bool`):
                指示输入 token 是否已经包含特殊 token。

        Returns:
            `List[int]`: 包含特殊 token 掩码的列表。
        """
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
        # If the token list already has special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # If token_ids_1 exists, create a mask for sequence pairs with special tokens
        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        # Otherwise, create a mask for a single sequence with special tokens
        return ([0] * len(token_ids_0)) + [1, 1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
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
        # Separator token for XLNet
        sep = [self.sep_token_id]
        # Class segment ID for XLNet
        cls_segment_id = [2]

        # If token_ids_1 is None, return mask for single sequence
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        
        # Otherwise, return mask for sequence pair
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id
    # 定义一个方法用于保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出词汇表文件路径，根据可选的前缀和默认的词汇表文件名组合而成
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件与输出文件不是同一个文件，并且当前词汇表文件存在，则进行复制操作
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的模型写入到输出文件中
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的输出词汇表文件路径的元组形式
        return (out_vocab_file,)
```