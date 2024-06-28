# `.\models\phobert\tokenization_phobert.py`

```py
# 指定编码格式为 UTF-8
# 版权声明，指出版权归属于 VinAI Research 和 HuggingFace Inc. 团队
# 版权声明，指出版权归属于 Open AI Team 作者和 HuggingFace Inc. 团队
#
# 根据 Apache License, Version 2.0 进行许可，除非符合许可要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，软件将按"原样"分发，不提供任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证
""" PhoBERT 的分词类 """

# 导入标准库中的 os 模块
# 导入标准库中的 re 模块，用于正则表达式操作
# 从 shutil 库中导入 copyfile 函数
# 从 typing 模块中导入 List、Optional 和 Tuple 类型
import os
import re
from shutil import copyfile
from typing import List, Optional, Tuple

# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类
# 从 utils 模块中导入 logging 函数
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 VOCAB_FILES_NAMES 字典，指定词汇和合并文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",       # 词汇文件名
    "merges_file": "bpe.codes",     # 合并文件名
}

# 定义 PRETRAINED_VOCAB_FILES_MAP 字典，指定预训练模型的词汇和合并文件的下载地址
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

# 定义 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 字典，指定预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vinai/phobert-base": 256,      # PhoBERT-base 的位置嵌入大小
    "vinai/phobert-large": 256,     # PhoBERT-large 的位置嵌入大小
}


def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号元组（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    pairs = set(pairs)
    return pairs


class PhobertTokenizer(PreTrainedTokenizer):
    """
    构造一个 PhoBERT 分词器。基于字节对编码（Byte-Pair-Encoding）。

    此分词器继承自 PreTrainedTokenizer，其中包含大多数主要方法。用户应参考这个超类以获取有关这些方法的更多信息。
    """
    # 定义一个类，用于处理包含特殊标记的词汇表和合并文件，以及各种特殊标记的设置
    
    vocab_files_names = VOCAB_FILES_NAMES
    # 将预定义的词汇表文件名映射保存到变量中
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预定义的预训练词汇表文件映射保存到变量中
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 将预定义的最大模型输入尺寸保存到变量中
    
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
    ):
    # 类的初始化方法，接受词汇表文件路径、合并文件路径和多个可选的特殊标记参数
    ):
        self.vocab_file = vocab_file  # 初始化词汇文件路径
        self.merges_file = merges_file  # 初始化合并文件路径

        self.encoder = {}  # 初始化编码器字典
        self.encoder[str(bos_token)] = 0  # 设置开始符号的编码为0
        self.encoder[str(pad_token)] = 1  # 设置填充符号的编码为1
        self.encoder[str(eos_token)] = 2  # 设置结束符号的编码为2
        self.encoder[str(unk_token)] = 3  # 设置未知符号的编码为3

        self.add_from_file(vocab_file)  # 从词汇文件中添加更多编码到编码器字典中

        self.decoder = {v: k for k, v in self.encoder.items()}  # 根据编码器字典创建解码器字典，用于反向查找

        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]  # 读取并分割合并文件内容为列表，排除最后的空行
        merges = [tuple(merge.split()[:-1]) for merge in merges]  # 将每行的合并操作转换为元组列表

        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # 创建BPE合并操作到排名的映射字典
        self.cache = {}  # 初始化缓存字典

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )  # 调用父类的初始化方法，设置特殊标记符号及其ID

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A PhoBERT sequence has the following format:

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
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]  # 返回单个序列的输入ID列表，包含特殊标记符号
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep  # 返回序列对的输入ID列表，包含特殊标记符号

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
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

        # Check if special tokens are already present in the token list
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token_ids_1 is None, return a list indicating special tokens at the beginning and end of token_ids_0
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # If token_ids_1 is provided, return a list with special tokens marking both sequences
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

        # Define special tokens for the beginning and end of sequences
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros for token type ids based on token_ids_0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If token_ids_1 is provided, return a list of zeros for token type ids based on both sequences
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Return the size of the vocabulary, which is the length of the encoder dictionary
        return len(self.encoder)

    def get_vocab(self):
        # Return a combined dictionary of the encoder and added_tokens_encoder
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        # 将 token 转换为元组形式的单词
        word = tuple(token)
        # 在单词末尾添加 "</w>" 标记，转换为新的元组形式的单词
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        # 获取单词中的所有可能的 bigram 组合
        pairs = get_pairs(word)

        # 如果没有 bigram 可用，则直接返回原始 token
        if not pairs:
            return token

        # 不断循环，直到无法再分割为 bigram
        while True:
            # 选取在 self.bpe_ranks 中排名最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果选取的 bigram 不在 self.bpe_ranks 中，则停止循环
            if bigram not in self.bpe_ranks:
                break
            # 分割单词，替换为新的单词形式
            first, second = bigram
            new_word = []
            i = 0
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
            # 如果单词长度为 1，则停止分割
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将单词中的 "@@ " 替换为空格，并移除末尾的特殊标记
        word = "@@ ".join(word)
        word = word[:-4]
        # 将处理后的结果存入缓存中
        self.cache[token] = word
        # 返回处理后的单词
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        split_tokens = []

        # 使用正则表达式将文本分割成单词列表
        words = re.findall(r"\S+\n?", text)

        # 对每个单词进行 BPE 分词，然后拆分成更小的子单词列表
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        
        # 返回分词后的子单词列表
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 获取其在词汇表中的 id，如果不存在则返回未知标记的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 id 获取其在词汇表中对应的 token，如果不存在则返回未知标记
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 列表拼接成单个字符串，并移除 BPE 分词时添加的特殊标记
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        # 返回拼接后的字符串
        return out_string
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
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

        # 如果当前词汇表文件路径与输出路径不同且当前文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将当前实例的sp_model序列化后写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 如果当前合并文件路径与输出路径不同，则复制当前合并文件到输出路径
        if os.path.abspath(self.merges_file) != os.path.abspath(out_merge_file):
            copyfile(self.merges_file, out_merge_file)

        # 返回输出的词汇表文件路径和合并文件路径
        return out_vocab_file, out_merge_file

    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)

    def add_from_file(self, f):
        """
        从文本文件加载预先存在的字典，并将其符号添加到此实例中。
        """
        # 如果参数f是字符串，则尝试打开文件并递归调用add_from_file
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(f"Incorrect encoding detected in {f}, please rebuild the dataset")
            return
        
        # 读取文件所有行
        lines = f.readlines()
        # 遍历每一行
        for lineTmp in lines:
            # 去除每行两端的空白字符
            line = lineTmp.strip()
            # 在行中查找最后一个空格的索引位置
            idx = line.rfind(" ")
            # 如果没有找到空格，则抛出值错误
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            # 获取单词部分
            word = line[:idx]
            # 将单词添加到实例的编码器中，使用当前编码器的长度作为值
            self.encoder[word] = len(self.encoder)
```