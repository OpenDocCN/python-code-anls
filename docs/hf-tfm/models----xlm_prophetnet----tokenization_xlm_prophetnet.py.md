# `.\models\xlm_prophetnet\tokenization_xlm_prophetnet.py`

```py
# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections  # 引入collections模块，用于数据结构的操作
import os  # 引入os模块，用于文件系统操作
from shutil import copyfile  # 引入shutil模块的copyfile函数，用于复制文件
from typing import Any, Dict, List, Optional, Tuple  # 引入常见数据类型定义
from ...tokenization_utils import PreTrainedTokenizer  # 引入预训练模型的文本分割工具
from ...utils import logging  # 引入库中通用的日志记录工具

# 初始化日志记录器
logger = logging.get_logger(__name__)

# 句子分片下划线符号（用于分词标识）
SPIECE_UNDERLINE = "▁"

# 预训练模型的各种名字——主要指的是词典文件名
VOCAB_FILES_NAMES = {"vocab_file": "prophetnet.tokenizer"}

# 预训练模型提供者映射词典及其模型位置
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/xprophetnet-large-wiki100-cased": (
            "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/prophetnet.tokenizer"
        ),
    }
}

# 预训练初始化配置参数
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/xprophetnet-large-wiki100-cased": {"do_lower_case": False},
}

# 预训练模型中预定义的position嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/xprophetnet-large-wiki100-cased": 512,
}

# 加载词典文件中的词汇
def load_vocab(vocab_file):
    """
    将词汇文件加载到字典中

    :param vocab_file: 词汇文件路径和名称
    :type vocab_file: str
    """
    vocab = collections.OrderedDict()  # 初始化排序后的词汇字典
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇文件
        tokens = reader.readlines()  # 读取文件所有内容
    for index, token in enumerate(tokens):  # 遍历每一个词汇及其索引
        token = token.rstrip("\n")  # 移除字符串尾部的换行符
        vocab[token] = index  # 将词汇添加到词汇字典中，并指派相应索引
    return vocab

# 定义用于处理各种语言模型的类 - XLMProphetNetTokenizer
# 该类继承自 PreTrainedTokenizer 类，并且:
# - 使用了 SentencePiece 技术解决了分词问题
# - 将文本转化为模型能够处理的序列
# - 含有用于辅助加载预训练模型参数的方法和属性
    # 定义一个函数，用于初始化一个词汇表的配置
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径

        bos_token (`str`, *optional*, defaults to `"[SEP]"`):
            # 序列开始的特殊标记，用于预训练。在构建序列时，实际使用的是 `cls_token`。

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            # 序列结束的特殊标记。在构建序列时，实际使用的是 `sep_token`。

        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            # 分隔标记，在构建多个序列的时候使用，例如序列分类或问答任务中的问题和文本分隔。

        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            # 未知标记。词汇表中不存在的标记将会被替换为该标记。

        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            # 填充标记，用于对不同长度的序列进行批处理时进行填充。

        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            # 分类器标记，在序列分类任务中，是构建序列时的第一个标记。

        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            # 掩码标记，用于掩码语言建模训练中，模型将尝试预测这些标记。

        sp_model_kwargs (`dict`, *optional*):
            # 传递给 `SentencePieceProcessor.__init__()` 方法的参数字典，用于设置 SentencePiece 模型的初始化参数。
            # 可用的参数包括 `enable_sampling`（启用子词正则化）、`nbest_size`（用于unigram的采样参数，对于BPE-Dropout无效）、
            # `alpha`（unigram采样的平滑参数和BPE-dropout的合并操作的dropout概率）等。
            # 参考 [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) 获取更多信息。
    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """



    # 定义类变量，指定了模型使用的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义类变量，指定了预训练模型使用的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义类变量，指定了预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义类变量，指定了模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]



    def __init__(
        self,
        vocab_file,
        bos_token="[SEP]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果没有提供 sp_model_kwargs 则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        try:
            # 尝试导入 sentencepiece 库
            import sentencepiece as spm
        except ImportError:
            # 如果导入失败，给出警告并提示用户安装 SentencePiece 库的链接和安装指令
            logger.warning(
                "You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiece"
                " pip install sentencepiece"
            )
            raise

        # 初始化 sp_model 属性，使用给定的 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到 sp_model
        self.sp_model.Load(str(vocab_file))
        # 保存词汇文件路径到 vocab_file 属性
        self.vocab_file = vocab_file

        # 原始 fairseq 的词汇和 spm 的词汇必须是“对齐”的:
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # 将特殊的 tokens 和 [unused] tokens 放入词汇表中
        self.fairseq_tokens_to_ids = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "[MASK]": 4}

        for i in range(10):
            tok = f"[unused{i}]"
            self.fairseq_tokens_to_ids[tok] = 5 + i

        # 第一个“真实”的 token “,” 在嵌入词汇中的位置为 15，在 spm 词汇中的位置为 3
        self.fairseq_offset = 12
        # 创建 fairseq_ids_to_tokens 字典，用于根据 id 查找对应的 token
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # TODO ArthurZ fairseq_ids_to_tokens should be removed

        # 调用父类的初始化方法，传入各种特殊 token 和 sp_model_kwargs 等参数
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
        # 检查词汇文件是否存在，从而确定是否可以保存慢速的分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False



    def __getstate__(self):
        # 返回对象的状态字典，将 sp_model 设为 None，以便对象可以被序列化
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state
    def __setstate__(self, d):
        self.__dict__ = d  # 将对象的属性字典设置为给定的字典 `d`

        try:
            import sentencepiece as spm  # 尝试导入 sentencepiece 库
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiece"
                " pip install sentencepiece"
            )
            raise  # 报错提醒用户需要安装 SentencePiece 库

        # 用于向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}  # 如果对象没有 `sp_model_kwargs` 属性，则设置为空字典

        # 根据 `self.sp_model_kwargs` 参数创建 SentencePieceProcessor 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
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
            )  # 如果已经包含特殊标记，调用父类的方法获取特殊标记掩码

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]  # 返回仅有第一个序列的特殊标记掩码
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]  # 返回包含两个序列的特殊标记掩码

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

        sep = [self.sep_token_id]  # 获取分隔符的 token id

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]  # 返回仅有第一个序列的 token type ids
        return len(token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 返回包含两个序列的 token type ids

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset  # 返回词汇表大小

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # 构建词汇表字典
        vocab.update(self.added_tokens_encoder)  # 添加额外的编码器信息到词汇表
        return vocab  # 返回词汇表字典
    def _tokenize(self, text: str) -> str:
        """Tokenizes a given text using the SentencePiece model and returns it as a string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into its corresponding ID using the vocabulary."""
        # Check if the token exists in the predefined Fairseq tokens to IDs mapping
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        # Obtain the token's ID from the SentencePiece model
        spm_id = self.sp_model.PieceToId(token)
        # Return the ID with an offset specific to Fairseq or the unknown token ID if not found
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into its corresponding token (str) using the vocabulary."""
        # Check if the index exists in the predefined Fairseq IDs to tokens mapping
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        # Convert the index to a token using the SentencePiece model adjusted by Fairseq offset
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (strings for sub-words) into a single concatenated string,
        replacing special sub-word marker with spaces and stripping leading/trailing spaces.
        """
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the current vocabulary to the specified directory.

        Args:
            save_directory (str): Directory path where the vocabulary file should be saved.
            filename_prefix (Optional[str]): Optional prefix for the vocabulary file name.

        Returns:
            Tuple[str]: Tuple containing the path of the saved vocabulary file.
        """
        # Ensure the provided directory path exists; otherwise, log an error
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # Define the output vocabulary file path based on the provided directory and filename prefix
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Copy the current vocabulary file if it differs from the destination path and exists
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If the current vocabulary file doesn't exist, write the serialized SentencePiece model to the output file
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds model inputs by concatenating a sequence or pair of sequences with special tokens.

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (Optional[List[int]]): Optional list of token IDs for the second sequence in a pair.

        Returns:
            List[int]: List of input IDs with added special tokens for model input.
        """
        # If only one sequence is provided, concatenate it with the separator token
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        # Concatenate both sequences with separator tokens in between
        sep = [self.sep_token_id]
        return token_ids_0 + sep + token_ids_1 + sep
```