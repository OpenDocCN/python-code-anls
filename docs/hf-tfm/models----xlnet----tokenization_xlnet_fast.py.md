# `.\transformers\models\xlnet\tokenization_xlnet_fast.py`

```
# 设置 Python 文件的字符编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则无法使用该文件
# 可以在以下网址获取许可证的拷贝
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得使用该软件
# 根据许可证分发的软件基于"现状"分发，没有任何明示或暗示的保证或条件
# 请参阅特定语言的许可证中对权限和限制的说明
# 适用于 XLNet 模型的标记化类
# 导入所需的模块和类
import os
from shutil import copyfile
from typing import List, Optional, Tuple
# 导入 tokenization_utils 模块中的 AddedToken 类
from ...tokenization_utils import AddedToken
# 导入 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 导入 utils 模块中的 is_sentencepiece_available 和 logging 方法
from ...utils import is_sentencepiece_available, logging

# 如果 SentencePiece 可用，则导入 tokenization_xlnet 模块中的 XLNetTokenizer 类，否则设置为 None
if is_sentencepiece_available():
    from .tokenization_xlnet import XLNetTokenizer
else:
    XLNetTokenizer = None

# 获取 logger
logger = logging.get_logger(__name__)

# 定义 XLNetTokenizerFast 类
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}
# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/spiece.model",
        "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/tokenizer.json",
        "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/tokenizer.json",
    },
}
# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlnet-base-cased": None,
    "xlnet-large-cased": None,
}
# 分隔符
SPIECE_UNDERLINE = "▁"
# 段落标识符
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

# 定义 XLNetTokenizerFast 类
class XLNetTokenizerFast(PreTrainedTokenizerFast):
    """
    构建"快速"的 XLNet 标记化器（由 HuggingFace 的 *tokenizers* 库支持）。基于 [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。
    该标记化器继承自 `PreTrainedTokenizerFast`，其中包含大多数主要方法。用户应参考该超类以获取更多关于这些方法的信息。
    # 定义一个类，用于实例化一个 tokenizer，具体参数如下：
    
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            <Tip>
                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the `cls_token`.
            </Tip>
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            <Tip>
                When building a sequence using special tokens, this is not the token that is used for the end of sequence.
                The token used is the `sep_token`.
            </Tip>
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.
    
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    
    # VOCAB_FILES_NAMES 是一个常量，用于存储文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 将预训练词汇文件映射赋值给变量
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练定位嵌入大小映射赋值给变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设定填充位置在左边
    padding_side = "left"
    # 慢速标记器类为 XLNetTokenizer

    # 初始化方法，包含多个参数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
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
        **kwargs,
    ):
        # 如果 mask_token 是字符串，添加的标记将表现得像普通单词，即包括其之前的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类初始化方法
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
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
            **kwargs,
        )

        # 设置 _pad_token_type_id 属性为 3
        self._pad_token_type_id = 3
        # 赋值其他属性
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    # 定义 can_save_slow_tokenizer 属性方法，用于判断是否可以保存慢速标记器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过串联和添加特殊标记，从一个序列或一个序列对构建用于序列分类任务的模型输入
        Args:
            token_ids_0 (`List[int]`): 将添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对

        Returns:
            `List[int]`: 包含适当的特殊标记的 [输入 ID](../glossary#input-ids) 列表
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果没有 token_ids_1，返回 token_ids_0 + sep + cls
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        # 否则返回 token_ids_0 + sep + token_ids_1 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    # 从序列中创建 token_type_ids 方法
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 该函数用于创建一个掩码(mask)，用于序列对分类任务
    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # XLNet 序列对掩码的格式如下:
        # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        # | first sequence    | second sequence |
        # 如果 token_ids_1 为 None, 则只返回第一部分的掩码(0s)
        
        # 定义特殊字符的 token ID
        sep = [self.sep_token_id]
        cls_segment_id = [2]
        
        # 如果只有一个序列, 则返回第一部分的掩码加上 cls_segment_id
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        # 如果有两个序列, 则返回两部分的掩码加上 cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id
    
    # 该函数用于保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果当前的快速 tokenizer 没有保存慢 tokenizer 所需的信息, 则抛出异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )
        
        # 如果保存目录不是一个目录, 则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构造输出词汇表文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 如果输出路径与当前的词汇表文件路径不同, 则复制文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        
        # 返回输出词汇表文件的路径
        return (out_vocab_file,)
```