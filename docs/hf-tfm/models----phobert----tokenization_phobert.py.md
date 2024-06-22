# `.\transformers\models\phobert\tokenization_phobert.py`

```py
# 设定文件编码为 utf-8
# 版权声明，引用 VinAI Research 和 HuggingFace Inc. 团队的版权声明
# 使用 Apache 许可证 2.0 版本授权
# 只能在符合许可证的前提下使用此文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据 "AS IS" 基础发布的软件，无任何保证或条件，无论是明示还是暗示
# 请参考许可证中关于特定语言的具体权限和限制
""" PhoBERT 的 Tokenization 类"""


# 导入必要库
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入预训练分词器
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取记录器实例
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "merges_file": "bpe.codes",
}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vinai/phobert-base": "https://huggingface.co/vinai/phobert-base/resolve/main/vocab.txt",
        "vinai/phobert-large": "https://huggingface.co/vinai/phobert-large/resolve/main/vocab.txt",
    },
    "merges_file": {
        "vinai/phobert-base": "https://huggingface.co/vinai/phobert-base/resolve/main/bpe.codes",
        "vinai/phobert-large": "https://huggingface.co/vinai/phobert-large/resolve/main/bpe.codes",
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vinai/phobert-base": 256,
    "vinai/phobert-large": 256,
}

# 定义获取符号对的函数
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs

# 定义 PhoBERT Tokenizer 类，继承自 PreTrainedTokenizer 类
class PhobertTokenizer(PreTrainedTokenizer):
    """
    Construct a PhoBERT tokenizer. Based on Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    # 这些是输入参数的注释，描述了每个参数的含义和用途
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        bos_token (`st`, *optional*, defaults to `"<s>"`):
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
    
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    
    # 以下是类定义的一部分
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    # 定义 PhoBERTConfig 类，包含词汇表文件、合并文件路径以及特殊 token 的编码
    def __init__(
        self,
        vocab_file,
        merges_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        cls_token="<s>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        # 设置词汇表文件和合并文件路径
        self.vocab_file = vocab_file
        self.merges_file = merges_file
    
        # 定义特殊 token 的编码
        self.encoder = {}
        self.encoder[str(bos_token)] = 0
        self.encoder[str(pad_token)] = 1
        self.encoder[str(eos_token)] = 2
        self.encoder[str(unk_token)] = 3
    
        # 从文件中添加词汇表
        self.add_from_file(vocab_file)
    
        # 创建解码器
        self.decoder = {v: k for k, v in self.encoder.items()}
    
        # 读取合并文件，创建 BPE 合并的字典
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:-1]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
    
        # 调用父类的构造函数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
    
    # 构建输入序列，添加特殊 token
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        构建用于序列分类任务的输入序列，添加特殊 token。
        PhoBERT 序列的格式如下:
        - 单个序列: `<s> X </s>`
        - 两个序列: `<s> A </s></s> B </s>`
        Args:
            token_ids_0 (`List[int]`):
                要添加特殊 token 的序列 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列 ID 列表。
        Returns:
            `List[int]`: 包含appropriate特殊 token 的输入 ID 列表。
        """
        if token_ids_1 is None:
            # 单个序列: 添加 cls 和 sep token
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        else:
            # 两个序列: 添加 cls、sep 和 sep token
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + sep + token_ids_1 + sep
    
    # 获取特殊 token 的 mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
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
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. PhoBERT does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            `int`: The size of the vocabulary.
        """
        return len(self.encoder)

    def get_vocab(self):
        """
        Get the vocabulary of the tokenizer.

        Returns:
            `dict`: A dictionary containing the encoder and added tokens encoder.
        """
        return dict(self.encoder, **self.added_tokens_encoder)
    # 对输入的 token 进行 BPE 编码
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 将 token 最后一个字符与 "</w>" 组合成新的元组
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        # 获取 token 中的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回 token
        if not pairs:
            return token

        # 循环处理字符对
        while True:
            # 找到当前字符对中频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果字符对不在 BPE 编码表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 根据字符对进行 BPE 编码
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果编码后的字符长度为 1，则跳出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将编码后的字符用 "@@" 连接起来
        word = "@@ ".join(word)
        # 去除末尾的 "@@"
        word = word[:-4]
        # 将结果存入缓存并返回
        self.cache[token] = word
        return word

    # 对输入的文本进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        split_tokens = []

        # 使用正则表达式找出文本中的单词
        words = re.findall(r"\S+\n?", text)

        # 对每个单词进行 BPE 编码并拆分
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        return split_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string
    # 保存词汇表到指定目录，可指定文件名前缀，默认返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建输出的合并文件路径
        out_merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的模型内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 如果当前合并文件路径与输出路径不同，则复制当前合并文件到输出路径
        if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
            copyfile(self.merges_file, out_merge_file)

        # 返回输出的词汇表文件路径和合并文件路径
        return out_vocab_file, out_merge_file

    # 从文件中加载预先存在的字典，并将其符号添加到该实例中
    def add_from_file(self, f):
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            return

        # 逐行读取文件内容
        lines = f.readlines()
        for lineTmp in lines:
            line = lineTmp.strip()
            # 查找最后一个空格的位置
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            # 提取单词和计数
            word = line[:idx]
            # ��单词添加到编码器中，编码器的值为当前编码器的长度
            self.encoder[word] = len(self.encoder)
```