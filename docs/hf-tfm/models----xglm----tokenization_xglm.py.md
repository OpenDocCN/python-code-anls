# `.\transformers\models\xglm\tokenization_xglm.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可版本 2.0 授权使用此文件
# 除非符合许可，否则不能使用此文件
# 可以在以下网址获取许可副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按 "原样" 分发软件
# 不提供任何形式的担保或条件，不论是明示的还是暗示的
# 有关版权声明下的具体语言和限制，请参见许可
"""Tokenization classes for ."""

# 导入必要的库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库
import sentencepiece as spm

# 导入 Hugging Face 库中的相关工具
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SentencePiece 分隔符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/sentencepiece.bpe.model"
    }
}

# 预训练位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/xglm-564M": 2048
}

# 定义 XGLMTokenizer 类，继承自 PreTrainedTokenizer
class XGLMTokenizer(PreTrainedTokenizer):
    """
    根据 RobertaTokenizer 和 XLNetTokenizer 进行调整，基于 SentencePiece。
    
    这个分词器继承自 PreTrainedTokenizer，其中包含大多数主要方法。用户应该参考
    这个超类获取有关这些方法的更多信息。
    """
    # 定义输入参数的注释信息
    Args:
        # 指定词汇表文件的路径
        vocab_file (`str`):
            Path to the vocabulary file.
        # 指定序列开始标记，默认为 "<s>"
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
            <Tip>
            当构建包含特殊标记的序列时，这不是用于序列开始的标记。使用的标记是 `cls_token`。
            </Tip>
        # 指定序列结束标记，默认为 "</s>"
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
    
            <Tip>
            当构建包含特殊标记的序列时，这不是用于序列结束的标记。使用的标记是 `sep_token`。
            </Tip>
        # 指定分隔标记，用于构建由多个序列组成的序列，如文本分类或问答任务中的文本和问题序列
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        # 指定分类标记，用于序列分类任务
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        # 指定未知标记，当输入的词不在词汇表中时使用该标记
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        # 指定填充标记，用于对不同长度的序列进行批处理时的填充
        pad_token (`str`, *optional`, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        # 指定额外的参数传递给 SentencePieceProcessor 初始化函数
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
        # 保存 SentencePieceProcessor 对象，用于各种转换(字符串、标记和 ID)
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    
    # 定义词汇表文件名和预训练词汇表文件映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义一个类BartTokenizer，继承自PreTrainedTokenizer，在构造函数中指定其各种特殊符号token以及其他参数
    class BartTokenizer(PreTrainedTokenizer):
        # 定义最大模型输入大小为预训练位置嵌入大小的列表
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # 定义模型输入的token名字列表
        model_input_names = ["input_ids", "attention_mask"]
    
        # 构造函数
        def __init__(
            self,
            vocab_file,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
        ) -> None:
            # 如果未提供sp_model_kwargs，则默认为空字典
            self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
    
            # 与原始的tokenizer兼容，定义了7个虚构的词汇
            self.num_madeup_words = 7
            madeup_words = [f"<madeupword{i}>" for i in range(self.num_madeup_words)]
    
            # 将额外的特殊token加入到additional_special_tokens列表中
            kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
            kwargs["additional_special_tokens"] += [
                word for word in madeup_words if word not in kwargs["additional_special_tokens"]
            ]
    
            # 使用SentencePieceProcessor加载vocab文件
            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.Load(str(vocab_file))
            self.vocab_file = vocab_file
    
            # 原始fairseq vocab和spm vocab必须"对齐"
            # 符号 "|" 之左侧是fairseq中的token，之右侧是spm中的token
            # 索引    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
            # ------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
            # fairseq | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
            # spm     | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'
    
            # 第一个"真实"的token","在fairseq vocab中的位置为4，在spm vocab中的位置为3
            self.fairseq_offset = 1
    
            # 建立fairseq的token到id的映射
            self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    
            # 添加虚构词的token到id的映射
            sp_size = len(self.sp_model)
            madeup_words = {f"<madeupword{i}>": sp_size + i + self.fairseq_offset for i in range(self.num_madeup_words)}
            self.fairseq_tokens_to_ids.update(madeup_words)
    
            # 建立fairseq的id到token的映射
            self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
    
            # 调用父类的构造函数初始化Tokenizer
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                sp_model_kwargs=self.sp_model_kwargs,
                **kwargs,
            )
    
        # 重写序列化对象的方法，返回需要存储的对象状态
        def __getstate__(self):
            state = self.__dict__.copy()
            state["sp_model"] = None
            state["sp_model_proto"] = self.sp_model.serialized_model_proto()
            return state
    
        # 重写反序列化对象的方法，将存储的对象状态赋值给当前对象
        def __setstate__(self, d):
            self.__dict__ = d
    
            # 向后兼容
            if not hasattr(self, "sp_model_kwargs"):
                self.sp_model_kwargs = {}
    
            # 重新加载SentencePieceProcessor对象
            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

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

        if token_ids_1 is None:
            # 如果只有一个输入序列，则添加起始特殊标记后返回结果
            return [self.sep_token_id] + token_ids_0
        sep = [self.sep_token_id]
        # 如果有两个输入序列，则添加起始、终止特殊标记后返回结果
        return sep + token_ids_0 + sep + sep + token_ids_1

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
            # 如果已经添加了特殊标记，直接返回特殊标记掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 如果只有一个输入序列，返回带有起始特殊标记的特殊标记掩码
            return [1] + ([0] * len(token_ids_0))
        # 如果有两个输入序列，返回带有起始、终止特殊标记的特殊标记掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，用于创建一个在序列对分类任务中使用的掩码。XLM-RoBERTa不使用token type ids，因此返回一个由零组成的列表。
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        sep = [self.sep_token_id]  # 创建一个包含特殊分隔符 token id 的列表

        # 如果没有传入token_ids_1，则返回由零组成的列表
        if token_ids_1 is None:
            return len(sep + token_ids_0) * [0]
        # 否则返回根据传入的序列长度计算的由零组成的列表
        return len(sep + token_ids_0 + sep + sep + token_ids_1) * [0]

    # 返回词汇表大小的属性
    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + self.num_madeup_words

    # 获取词汇表方法
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # 创建词汇表
        vocab.update(self.added_tokens_encoder)  # 更新词汇表
        return vocab  # 返回词汇表

    # 将文本标记化的方法
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)  # 使用sp_model对文本进行编码，返回列表

    # 将标记转换为 id 的方法
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:  # 如果标记存在于fairseq_tokens_to_ids中
            return self.fairseq_tokens_to_ids[token]  # 返回对应的id
        spm_id = self.sp_model.PieceToId(token)  # 获取标记对应的id

        # 如果SP模型返回0，需要返回未知标记
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    # 将id转换为标记的方法
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:  # 如果index存在于fairseq_ids_to_tokens中
            return self.fairseq_ids_to_tokens[index]  # 返回对应的标记
        return self.sp_model.IdToPiece(index - self.fairseq_offset)  # 返回id对应的标记

    # 将标记序列转换为字符串的方法
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()  # 将标记序列连接为单个字符串
        return out_string  # 返回处理后的字符串

    # 保存词汇表到文件的方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):  # 如果save_directory不是目录
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")  # 记录错误日志
            return  # 返回空
        out_vocab_file = os.path.join(  # 拼接输出词汇文件路径
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇文件路径存在并且和输出路径不同，则复制词汇文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇文件不存在，则写入SP模型的内容
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)  # 返回输出的词汇文件路径
```