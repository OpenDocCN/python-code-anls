# `.\models\fnet\tokenization_fnet.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2021年由 Google Research、Google AI、Google Brain 和 HuggingFace 团队共同拥有
# 根据 Apache 许可证2.0版（“许可证”）授权；除非符合许可证，否则不得使用此文件
# 您可以在以下网址获得许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，基于许可证分发的软件是基于“原样”基础分发的，不带任何形式的担保或条件，无论是明示或暗示的
# 请参考许可证以获取有关权限和限制的特定语言

# 导入必要的库
# 从 shutil 库中导入拷贝文件的函数 copyfile
# 从 typing 库导入必要的类型
# 导入 sentencepiece 库进行分词
# 从 tokenization_utils 模块导入 AddedToken 和 PreTrainedTokenizer 类
# 导入日志记录功能
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名，主要包含一个名为"vocab_file"的文件
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练模型的词汇文件映射，可以根据预训练模型字符串名获取相应的词汇文件下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/spiece.model",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/spiece.model",
    },
}

# 预训练模型的位置嵌入大小映射，根据模型名称获取相应的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/fnet-base": 512,
    "google/fnet-large": 512,
}

# 定义 SentencePiece 使用的特殊字符
SPIECE_UNDERLINE = "▁"

# 定义 FNetTokenizer 类，继承自 PreTrainedTokenizer 类
# 适用于 FNet 模型的 tokenizer，基于 SentencePiece 实现
# 用户可以参考 PreTrainedTokenizer 超类以获取更多关于方法的信息
class FNetTokenizer(PreTrainedTokenizer):
    """
    Construct an FNet tokenizer. Adapted from [`AlbertTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece). This tokenizer inherits from [`PreTrainedTokenizer`]
    which contains most of the main methods. Users should refer to this superclass for more information regarding those
    methods.
``` 
    # 参数:
    # vocab_file (`str`):
    #   SentencePiece 文件（通常具有 *.spm* 扩展名），包含实例化分词器所需的词汇表。
    # do_lower_case (`bool`, *optional*, 默认为 `False`):
    #   是否在分词时将输入转换为小写。
    # remove_space (`bool`, *optional*, 默认为 `True`):
    #   是否在分词时去除文本中的空格（在分词时去除字符串前后的多余空格）。
    # keep_accents (`bool`, *optional*, 默认为 `True`):
    #   是否在分词时保留重音符号。
    # unk_token (`str`, *optional*, 默认为 `"<unk>"`):
    #   未知标记。词汇表中没有的标记无法转换为 ID，并被设置为此标记。
    # sep_token (`str`, *optional*, 默认为 `"[SEP]"`):
    #   分隔符标记，用于从多个序列构建序列时使用，例如用于序列分类或用于文本和问题进行问答。它也用作使用特殊标记构建的序列的最后一个标记。
    # pad_token (`str`, *optional*, 默认为 `"<pad>"`):
    #   用于填充的标记，例如在对不同长度的序列进行批处理时使用。
    # cls_token (`str`, *optional*, 默认为 `"[CLS]"`):
    #   分类器标记，用于进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）。它是使用特殊标记构建的序列的第一个标记。
    # mask_token (`str`, *optional*, 默认为 `"[MASK]"`):
    #   用于屏蔽值的标记。这是使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
    # sp_model_kwargs (`dict`, *optional*):
    #   将传递给 `SentencePieceProcessor.__init__()` 方法。可以用 [SentencePiece 的 Python 包装器](https://github.com/google/sentencepiece/tree/master/python) 设置：

    #   - `enable_sampling`：启用子词正则化。
    #   - `nbest_size`：unigram 的采样参数。对于 BPE-Dropout 无效。

    #     - `nbest_size = {0,1}`：不执行采样。
    #     - `nbest_size > 1`：从 nbest_size 结果中进行采样。
    #     - `nbest_size < 0`：假定 nbest_size 为无穷大，并使用前向过滤和后向采样算法从所有假设（格）中进行采样。
    #   - `alpha`：unigram 采样的平滑参数，以及 BPE-dropout 的合并操作的丢失概率。
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 定义默认的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练词汇文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练位置嵌入大小的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "token_type_ids"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果 mask_token 是字符串类型，那么创建一个去除左侧空格的特殊标记
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 如果 cls_token 是字符串类型，那么创建一个特殊标记
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串类型，那么创建一个特殊标记
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 mask_token 是字符串类型，那么创建一个特殊标记
        mask_token = AddedToken(mask_token, special=True) if isinstance(mask_token, str) else mask_token
        # 如果 sp_model_kwargs 为 None，则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 初始化参数
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 sp_model_kwargs 初始化 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(vocab_file)

        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.sp_model)

    def get_vocab(self):
        # 创建词汇表并更新添加的特殊标记
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 创建对象的状态副本，并将 sp_model 设置为 None
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 将对象的状态更新为给定的状态
        self.__dict__ = d

        # 兼容旧版本
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 初始化 SentencePieceProcessor 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
    # 对输入文本进行预处理，包括去除空格、替换特殊字符等操作
    def preprocess_text(self, inputs):
        # 如果需要去除空格，则去除输入文本首尾的空格并用空格连接剩余部分
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            # 否则，不做任何处理
            outputs = inputs
        # 替换文本中的特殊字符“``”和“''”为双引号
        outputs = outputs.replace("``", '"').replace("''", '"')

        # 如果不保留重音符号，则利用unicodedata库将文本中的重音符号进行规范化处理
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            # 过滤文本中的重音符号
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        # 如果需要将所有字符转换为小写，则将文本全部转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        # 返回预处理后的文本
        return outputs

    # 对文本进行分词处理
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 调用预处理函数对文本进行预处理
        text = self.preprocess_text(text)
        # 利用sp_model对文本进行编码处理得到pieces
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        # 遍历编码后的pieces进行处理
        for piece in pieces:
            # 如果piece长度大于1且最后一个字符为逗号并且倒数第二个字符为数字，则进行特殊处理
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 对piece进行处理并添加到new_pieces中
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

        # 返回经过处理后的pieces
        return new_pieces

    # 将token转换为其对应的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    # 将id转换为其对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    # 将一系列token组合成一个字符串
    # 从transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string复制而来
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊字符不使用sentencepiece模型进行解码
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
        # 返回拼接后的字符串
        return out_string.strip()

    # 解码token_ids得到文本
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # 用于将模型的输出转换为文本，调用父类的_decode方法
        text = super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )
        # 模仿Rust分词器的行为：在<unk>后面没有空格
        if not spaces_between_special_tokens:
            text = text.replace("<unk> ", "<unk>")
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊令牌从序列或序列对构建模型输入，用于序列分类任务。一个FNet序列的格式如下:
        
        - 单个序列: `[CLS] X [SEP]`
        - 序列对: `[CLS] A [SEP] B [SEP]`
        
        Args:
            token_ids_0 (`List[int]`):
                用于添加特殊令牌的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的用于序列对的第二个ID列表。
        
        Returns:
            `List[int]`: 带有适当特殊令牌的[输入ID](../glossary#input-ids)列表。
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
        从没有添加特殊令牌的令牌列表中检索序列ID。当使用分词器`prepare_for_model`方法添加特殊令牌时，将调用此方法。
        
        Args:
            token_ids_0 (`List[int]`):
                ID列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的用于序列对的第二个ID列表。
            already_has_special_tokens (`bool`, *optional*, 默认为`False`):
                令牌列表是否已经被格式化为模型的特殊令牌。

        Returns:
            `List[int]`: 一个整数列表，在范围[0, 1]内: 1表示特殊令牌，0表示序列令牌。
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
   # 定义函数的返回类型为 List[int]
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An FNet sequence
        pair mask has the following format: :

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second sequence |
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
        # 创建包含分隔符token id的列表
        sep = [self.sep_token_id]
        # 创建包含类token id的列表
        cls = [self.cls_token_id]

        # 如果token_ids_1为空，返回只包含0的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回两个列表长度相加的0的列表和第二个列表的长度相加的1的列表
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果目录不存在，输出错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 定义输出词汇表文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件路径不同且存在，复制词汇表文件到输出文件路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，使用序列化模型内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```