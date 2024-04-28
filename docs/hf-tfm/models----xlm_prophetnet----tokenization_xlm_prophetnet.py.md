# `.\transformers\models\xlm_prophetnet\tokenization_xlm_prophetnet.py`

```py
# 编码声明，使用 UTF-8 编码
# 版权声明，告知代码的版权信息和许可协议
import collections
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger
logger = logging.get_logger(__name__)

# 定义下划线特殊字符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "prophetnet.tokenizer"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/xprophetnet-large-wiki100-cased": (
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer"
        ),
    }
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/xprophetnet-large-wiki100-cased": {"do_lower_case": False},
}

# 预训练模型的位置嵌入尺寸大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/xprophetnet-large-wiki100-cased": 512,
}

# 加载词汇表的函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 创建一个有序字典用于存储词汇表
    vocab = collections.OrderedDict()
    # 打开词汇表文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇表内容
        tokens = reader.readlines()
    # 将词汇表中的 token 填充到 vocab 中，并用索引号标注
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    # 返回填充好的 vocab
    return vocab

# XLMProphetNetTokenizer 类，继承于 PreTrainedTokenizer
class XLMProphetNetTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
```                                                                                                                                                                                                                                                           
    # 定义初始化 tokenizer 时的各种可选参数
    Args:
        # 词汇表文件的路径
        vocab_file (`str`):
            Path to the vocabulary file.
        # 开始序列标记，默认为 "[SEP]"
        bos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
            <Tip>
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
            </Tip>
        # 结束序列标记，默认为 "[SEP]"
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.
    
            <Tip>
            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.
            </Tip>
        # 分隔标记，默认为 "[SEP]"
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        # 未知标记，默认为 "[UNK]"
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        # 填充标记，默认为 "[PAD]"
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        # 分类标记，默认为 "[CLS]"
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        # 掩码标记，默认为 "[MASK]"
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        # 其他用于 SentencePiece 初始化的参数
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:
    
            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
    
              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
    
            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """
    
    # 这里定义了一些属性
    
    vocab_files_names = VOCAB_FILES_NAMES  # 词汇文件名字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练的词汇文件名字典
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 模型输入的最大尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 输入的模型名称列表
    
    def __init__(
        self,
        vocab_file,  # 词汇文件路径
        bos_token="[SEP]",  # 开始标志
        eos_token="[SEP]",  # 结束标志
        sep_token="[SEP]",  # 分隔标志
        unk_token="[UNK]",  # 未知标志
        pad_token="[PAD]",  # 填充标志
        cls_token="[CLS]",  # 类别标志
        mask_token="[MASK]",  # 掩码标志
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs  # SentencePiece 模型的相关参数
    
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiece"
                " pip install sentencepiece"
            )
            raise
    
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)  # 创建 SentencePieceProcessor 对象
        self.sp_model.Load(str(vocab_file))  # 加载词汇文件
        self.vocab_file = vocab_file  # 保存词汇文件路径
    
        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'
    
        # put special tokens and [unused] tokens into the vocab
        self.fairseq_tokens_to_ids = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}  # fairseq 的特殊 token 到 ID 的映射
    
        for i in range(10):
            tok = f"[unused{i}]"
            self.fairseq_tokens_to_ids[tok] = 5 + i  # fairseq 中 [unused] token 到 ID 的映射
    
        # The first "real" token "," has position 15 in the embedding vocab and position 3 in the spm vocab
        self.fairseq_offset = 12  # fairseq 中词汇表中真正的 token 的偏移位置
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}  # fairseq 的 ID 到 token 的映射
    
        # TODO ArthurZ fairseq_ids_to_tokens should be removed
    
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
    
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False  # 检查是否可以保存缓慢的 tokenizer
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state
    def __setstate__(self, d):
        # 用传入的字典更新对象的状态
        self.__dict__ = d
        try:
            # 尝试导入 sentencepiece 库
            import sentencepiece as spm
        except ImportError:
            # 如果导入失败，记录警告信息并抛出导入错误
            logger.warning(
                "You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiece"
                " pip install sentencepiece"
            )
            raise

        # 用于向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 sentencepiece 处理器对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(self.vocab_file)

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

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLMProphetNet
        does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """

        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # 返回词汇大小
        return len(self.sp_model) + self.fairseq_offset

    def get_vocab(self):
        # 获取词汇表
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    def _tokenize(self, text: str) -> str:
        # 使用sp_model对文本进行编码，并返回字符串形式的结果
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果token在fairseq_tokens_to_ids中，直接返回对应的ID
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # 否则使用sp_model将token转换成ID
        spm_id = self.sp_model.PieceToId(token)

        # 如果SP模型返回0，需要返回未知token的ID
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果index在fairseq_ids_to_tokens中，直接返回对应的token
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # 否则使用sp_model将index转换成token
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 将一系列token转换成单个字符串，并将子词的下划线替换为空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件的路径和文件名前缀
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件的绝对路径与目标文件路径不同，并且当前词汇表文件存在，则进行拷贝
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将sp_model的序列化模型写入目标文件
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
        adding special tokens. A XLMProphetNet sequence has the following format:

        - single sequence: `X [SEP]`
        - pair of sequences: `A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # 如果只有token_ids_0，则在末尾添加<sep> token的ID
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        sep = [self.sep_token_id]
        # 否则按照格式连接两个token序列，并添加<sep> token的ID
        return token_ids_0 + sep + token_ids_1 + sep
```