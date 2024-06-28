# `.\models\bert_generation\tokenization_bert_generation.py`

```
# coding=utf-8
# 上面的注释声明了编码格式和版权信息

# 导入所需的库和模块
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，用于处理分词任务
import sentencepiece as spm

# 导入日志模块，用于记录和输出日志信息
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 定义词汇文件的名称常量
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert_for_seq_generation": (
            "https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder/resolve/main/spiece.model"
        ),
    }
}

# 定义预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"bert_for_seq_generation": 512}

# 定义 BertGenerationTokenizer 类，继承自 PreTrainedTokenizer
class BertGenerationTokenizer(PreTrainedTokenizer):
    """
    Construct a BertGeneration tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """

# 代码块结束
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        sep_token (`str`, *optional*, defaults to `"<::::>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
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

    """
    # 定义了一些常量用于指定预训练模型所需的文件名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义了一个映射，指定了预训练模型所需的词汇文件的位置
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义了一个映射，指定了预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化一个空列表，用于存储前缀 tokens
    prefix_tokens: List[int] = []
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<::::>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        初始化函数，用于实例化一个新的 Tokenizer 对象。

        Parameters:
            vocab_file (str):
                SentencePiece 文件名，包含用于实例化分词器所需的词汇表。
            bos_token (str, optional, default="<s>"):
                序列的开始标记。
            eos_token (str, optional, default="</s>"):
                序列的结束标记。
            unk_token (str, optional, default="<unk>"):
                未知标记。如果词汇表中不存在的 token，将被设置为此标记。
            pad_token (str, optional, default="<pad>"):
                用于填充的标记，例如在对不同长度的序列进行批处理时使用。
            sep_token (str, optional, default="<::::>"):
                分隔符标记，用于构建由多个序列组成的序列，例如用于序列分类或问答时的文本和问题。
                也作为包含特殊标记的序列的最后一个标记使用。
            sp_model_kwargs (dict, optional):
                传递给 `SentencePieceProcessor.__init__()` 方法的参数字典。
                可以用于设置 SentencePiece 的各种参数，例如启用子词正则化等。
        """
    ) -> None:
        """初始化函数，用于设置特定的参数并加载SentencePiece模型。

        Args:
            sp_model_kwargs (dict, optional): SentencePiece模型的参数设置，默认为空字典。
            vocab_file (str): SentencePiece模型的词汇文件路径。
            bos_token (str, optional): SentencePiece模型的开始符号，默认为None。
            eos_token (str, optional): SentencePiece模型的结束符号，默认为None。
            unk_token (str, optional): SentencePiece模型的未知符号，默认为None。
            pad_token (str, optional): SentencePiece模型的填充符号，默认为None。
            sep_token (str, optional): SentencePiece模型的分隔符号，默认为None。
            **kwargs: 其他可能的参数。
        """
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file

        # 使用给定的参数初始化SentencePieceProcessor对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载指定的词汇文件到SentencePieceProcessor对象中
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法，设置特殊符号的参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """获取当前SentencePiece模型的词汇大小（词汇量）。"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """返回一个词汇表字典，将token转换为对应的id。"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 添加额外的特殊token编码到词汇表字典中
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        """获取对象的状态信息，用于序列化对象。"""
        state = self.__dict__.copy()
        state["sp_model"] = None  # 将sp_model设置为None，以便序列化时不包含模型
        return state

    def __setstate__(self, d):
        """设置对象的状态信息，用于反序列化对象。"""
        self.__dict__ = d

        # 为了向后兼容性，如果不存在sp_model_kwargs属性，则设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建SentencePieceProcessor对象并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """将输入的文本进行分词（tokenize），返回token列表。"""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """将token转换为对应的id。"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """将id转换为对应的token。"""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """将token序列转换为单个字符串。"""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊token不使用sentencepiece模型进行解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()
    # 保存词汇表到指定目录，可选择添加前缀到文件名
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出词汇表文件的路径，如果有前缀则添加到文件名中
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件与输出文件不是同一个文件并且当前词汇表文件存在，则复制当前词汇表文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的模型内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的输出词汇表文件路径的元组
        return (out_vocab_file,)
```