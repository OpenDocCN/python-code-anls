# `.\transformers\models\big_bird\tokenization_big_bird.py`

```
# 设置脚本编码为 UTF-8
# 版权声明，版权归 Google Research 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证要求或书面同意，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不附带任何形式的担保或条件，也没有对特定目的的隐含担保或条件。
# 请参阅许可证了解特定语言的权限和限制。
"""BigBird 的标记化类。"""


# 导入必要的库
import os  # 导入操作系统模块
import re  # 导入正则表达式模块
from shutil import copyfile  # 导入文件拷贝函数
from typing import Any, Dict, List, Optional, Tuple  # 导入类型提示模块

# 导入 sentencepiece 库，该库用于分词
import sentencepiece as spm

# 导入父类 PreTrainedTokenizer 和其他必要的模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入日志模块

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型对应的词汇文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/spiece.model",
        "google/bigbird-roberta-large": (
            "https://huggingface.co/google/bigbird-roberta-large/resolve/main/spiece.model"
        ),
        "google/bigbird-base-trivia-itc": (
            "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/spiece.model"
        ),
    }
}

# 定义预训练模型对应的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/bigbird-roberta-base": 4096,
    "google/bigbird-roberta-large": 4096,
    "google/bigbird-base-trivia-itc": 4096,
}


# 定义 BigBirdTokenizer 类，继承自 PreTrainedTokenizer 类
class BigBirdTokenizer(PreTrainedTokenizer):
    """
    构造一个 BigBird 标记器。基于 SentencePiece。

    此标记器继承自 `PreTrainedTokenizer`，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
```  
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
            用于实例化分词器的词汇表文件（通常具有 *.spm* 扩展名）。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。不在词汇表中的标记无法转换为ID，并被设置为此标记。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The begin of sequence token.
            序列的开始标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            序列的结束标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在对不同长度的序列进行批处理时。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔标记，在从多个序列构建序列时使用，例如用于序列分类的两个序列或用于问答的文本和问题。它也被用作使用特殊标记构建的序列的最后一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            用于屏蔽值的标记。在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            序列分类时使用的分类器标记（对整个序列进行分类而不是对每个标记进行分类）。当使用特殊标记构建时，它是序列的第一个标记。
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
            将传递给 `SentencePieceProcessor.__init__()` 方法。[SentencePiece 的 Python 封装](https://github.com/google/sentencepiece/tree/master/python) 可用于设置：

            - `enable_sampling`: 启用子词正则化。
            - `nbest_size`: 一元采样的采样参数。对于 BPE-Dropout 无效。

              - `nbest_size = {0,1}`: 不执行采样。
              - `nbest_size > 1`: 从 nbest_size 结果中进行采样。
              - `nbest_size < 0`: 假设 nbest_size 为无穷大，并使用前向-后向采样算法从所有假设（网格）中进行采样。

            - `alpha`: 一元采样的平滑参数，以及 BPE-dropout 的合并操作的丢弃概率。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: List[int] = []
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果 bos_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果 cls_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串，则创建一个 AddedToken 对象，不去除左右空格
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        # mask_token 像普通单词一样，包括其前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则设为一个空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 保存词汇文件路径
        self.vocab_file = vocab_file

        # 创建 SentencePieceProcessor 对象并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类初始化方法
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        # 获取词汇表，包括已添加的特殊标记
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 返回对象的状态，排除 sp_model，以免序列化时包含大量数据
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 设置对象的状态，并重新加载 SentencePieceProcessor
        self.__dict__ = d

        # 为了向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新加载 SentencePieceProcessor
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # 对文本进行分词处理，返回一个字符串列表
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为其在词汇表中的 id
        return self.sp_model.piece_to_id(token)
    # 将索引（整数）转换为标记（字符串）使用词汇表
    def _convert_id_to_token(self, index):
        # 使用 SentencePiece 模型将索引转换为标记
        token = self.sp_model.IdToPiece(index)
        return token

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string 复制而来
    # 将一系列标记（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 用于存储当前子标记
        current_sub_tokens = []
        # 用于存储输出字符串
        out_string = ""
        # 用于标记前一个标记是否是特殊标记
        prev_is_special = False
        # 遍历每个标记
        for token in tokens:
            # 确保特殊标记不使用 SentencePiece 模型解码
            if token in self.all_special_tokens:
                # 如果前一个标记不是特殊标记，则添加空格
                if not prev_is_special:
                    out_string += " "
                # 使用 SentencePiece 模型解码当前子标记，并添加到输出字符串中
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # 标记当前标记为特殊标记
                prev_is_special = True
                # 重置当前子标记列表
                current_sub_tokens = []
            else:
                # 将当前标记添加到当前子标记列表中
                current_sub_tokens.append(token)
                # 标记当前标记不是特殊标记
                prev_is_special = False
        # 使用 SentencePiece 模型解码剩余的子标记，并添加到输出字符串中
        out_string += self.sp_model.decode(current_sub_tokens)
        # 返回去除首尾空格的输出字符串
        return out_string.strip()

    # 将标记 ID 列表解码为文本
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        # 从参数中取出是否使用源标记器的标志，并从参数中删除该项
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 将 token_ids 转换为对应的 tokens，并过滤掉特殊 token
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 为了避免混合字节级和 Unicode，我们需要分别构建字符串，用于添加的 tokens 和字节级 tokens
        # 参考：https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            # 跳过特殊 token 如果 skip_special_tokens 为真
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 如果 token 是添加的 token
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 模仿 Rust 标记器的行为：
        # [MASK] 和 [SEP] 前不加空格
        if spaces_between_special_tokens:
            text = re.sub(r" (\[(MASK|SEP)\])", r"\1", " ".join(sub_texts))
        else:
            text = "".join(sub_texts)

        # 根据 clean_up_tokenization_spaces 参数决定是否清理标记化空格
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            # 清理标记化空格并返回清理后的文本
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            # 否则返回文本
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则报错并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇文件不同于输出文件且存在，则复制词汇文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇文件不存在，则将 sp_model 序列化后的内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出文件路径的元组
        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 Big Bird 序列的格式如下：

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
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
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

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second
        sequence | If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 用于分隔两个序列的标记
        sep = [self.sep_token_id]
        # 用于表示序列的开始的标记
        cls = [self.cls_token_id]
        # 如果第二个序列不存在，返回只包含第一个序列的 mask，即全为 0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回包含两个序列的 mask，第一个序列对应的 token type ID 为 0，第二个序列对应的 token type ID 为 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
```