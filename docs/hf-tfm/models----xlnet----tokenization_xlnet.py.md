# `.\transformers\models\xlnet\tokenization_xlnet.py`

```py
# 导入所需库和模块
import os  # 用于操作系统相关功能
import unicodedata  # 用于 Unicode 字符处理
from shutil import copyfile  # 用于复制文件
from typing import Any, Dict, List, Optional, Tuple  # 用于类型提示

import sentencepiece as spm  # 导入 sentencepiece 库，用于分词

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入基础的 Tokenizer 类和添加的 Token 类
from ...utils import SPIECE_UNDERLINE, logging  # 导入 SPIECE_UNDERLINE 和 logging 相关内容

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义词汇文件名常量
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/spiece.model",
        "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/spiece.model",
    }
}

# 定义预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlnet-base-cased": None,
    "xlnet-large-cased": None,
}

# 定义 XLNet tokenizer 类
class XLNetTokenizer(PreTrainedTokenizer):
    """
    Construct an XLNet tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 定义词汇文件名和预训练模型词汇文件映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    padding_side = "left"

    # 初始化 XLNet tokenizer
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
        additional_special_tokens=["
    # 定义了一个函数，用来初始化一个新的tokenizer对象
    def __init__(
        self, vocab_file, merges_file, errors="replace", special_tokens=None, verbose=True,
        bos_token=self.bos_token, eos_token=self.eos_token, unk_token=self.unk_token,
        sep_token=self.sep_token, pad_token=self.pad_token, cls_token=self.cls_token, mask_token=self.mask_token,
        additional_special_tokens=self.additional_special_tokens, sp_model_kwargs=None,
    ) -> None:
        # 如果mask_token是字符串，那么将其作为特殊字符处理，否则保持原样
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 如果sp_model_kwargs为空，则创建一个空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 初始化一些实例变量
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        # 根据给定的参数创建一个SentencePieceProcessor对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从给定的vocab_file中加载模型
        self.sp_model.Load(vocab_file)
        # 调用父类的构造函数进行初始化
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
        # 初始化_pad_token_type_id变量为3
        self._pad_token_type_id = 3

    # 返回vocab的大小
    @property
    def vocab_size(self):
        return len(self.sp_model)

    # 获取vocab并返回一个包含词汇表的字典
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 获取当前对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 设定当前对象的状态
    def __setstate__(self, d):
        self.__dict__ = d
        # 为了向后兼容，如果不存在sp_model_kwargs，则将其初始化为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        # 用给定的sp_model_kwargs参数创建一个SentencePieceProcessor对象，然后加载���型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 对输入文本进行预处理，包括去除空格、替换引号、处理重音符号、转换为小写等
    def preprocess_text(self, inputs):
        # 如果指定要去除空格，则去除输入文本中的空格
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        # 替换一些特殊字符
        outputs = outputs.replace("``", '"').replace("''", '"')
        # 如果不保留重音符号，则将输出进行规范化，并删除其中的重音符号
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        # 如果需要转换为小写，则将输出文本转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        # 返回最终处理后的文本
        return outputs
    # 将文本分词成单词列表
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 预处理文本
        text = self.preprocess_text(text)
        # 使用句子处理模型对文本进行编码，得到分词后的结果
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        # 遍历分词后的结果
        for piece in pieces:
            # 处理分词结果，将逗号+数字的情况处理为独立的子词
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                # 进一步处理子词开头结尾的情况
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                # 将处理后的子词加入到新的分词结果列表中
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        # 返回处理后的分词结果
        return new_pieces

    # 将单词转换为对应的 ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    # 将 ID 转换为对应的单词
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    # 将 token 列表转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # 解码操作，将 ID 列表解码为文本
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    # 定义一个方法，返回字符串类型
    def __call__(self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: Optional[bool] = None,) -> str:
        # 使用kwargs.pop方法获取并删除"use_source_tokenizer"参数的值，默认为False
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 转换token_ids中的id为特殊token，并过滤特殊token
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 遍历过滤后的token
        sub_texts = []  # 存储子文本
        current_sub_text = []  # 存储当前子文本
        for token in filtered_tokens:
            # 如果需要跳过特殊token并且当前token是特殊token，继续下一次循环
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 如果当前token在添加token编码器中
            if token in self.added_tokens_encoder:
                # 如果当前子文本不为空，将其转换为字符串并添加到sub_texts列表中，然后重置当前子文本
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)  # 将当前token添加到sub_texts列表中
            else:
                current_sub_text.append(token)  # 添加当前token到当前子文本中
        # 如果当前子文本不为空，将其转换为字符串并添加到sub_texts列表中
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 重新构建文本，没有特殊token之间没有空格
        text = "".join(sub_texts)

        # 如果传入参数clean_up_tokenization_spaces不为空，则使用传入的值，否则使用clean_up_tokenization_spaces的默认值
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        # 如果需要清理token化空格，则执行清理操作并返回结果
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    # 定义一个方法，用于构建带有特殊token的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLNet sequence has the following format:

        - single sequence: `X <sep> <cls>`
        - pair of sequences: `A <sep> B <sep> <cls>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]  # 分隔符的token id列表
        cls = [self.cls_token_id]  # 类别分割符的token id列表
        # 如果token_ids_1为空，则返回token_ids_0 + 分隔符 + 类别分割符
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        # 如果token_ids_1不为空，则返回token_ids_0 + 分隔符 + token_ids_1 + 分隔符 + 类别分割符
        return token_ids_0 + sep + token_ids_1 + sep + cls

    # 定义一个方法，用于获取特殊token的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        # 如果已经包含特殊标记，则调用父类的get_special_tokens_mask方法

        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1, 1]
        # 如果有第二个序列ids，则返回第一个ids列表后加1，再加第二个ids列表再加1，最后加两个1

        return ([0] * len(token_ids_0)) + [1, 1]
        # 如果只有一个ids列表，则返回ids列表后加两个1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        # 如果token_ids_1为空，则返回第一个ids列表加上sep的长度的0，并添加一个cls_segment_id

        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id
        # 如果token_ids_1不为空，则返回第一个ids列表加上sep的长度的0，再加上第二个ids列表加上sep的长度的1，并添加一个cls_segment_id
    # 将词汇表保存到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查输入的保存目录是否为目录
        if not os.path.isdir(save_directory):
            # 如果不是目录, 记录错误日志并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出文件的完整路径, 包括前缀(如果提供)和默认文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
    
        # 如果当前词汇表文件路径和输出路径不同, 且当前词汇表文件存在, 则将其复制到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在, 则将序列化的词汇表模型内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
    
        # 返回输出文件的完整路径
        return (out_vocab_file,)
```