# `.\models\deprecated\transfo_xl\tokenization_transfo_xl.py`

```
# 设置文件编码为utf-8
# 版权声明，列举了作者和团队，以及NVIDIA CORPORATION的版权
# 根据 Apache License, Version 2.0 许可获取许可的链接
# License 规定了使用此文件的条件
# 导入所需模块
"""
 Tokenization classes for Transformer XL model. Adapted from https://github.com/kimiyoung/transformer-xl.
"""

# 导入必要的库和工具
# 以下部分是从指定地址导入并处理数据的代码片段
import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
# 导入numpy库
import numpy as np

# 导入相应函数和工具
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
    cached_file,
    is_sacremoses_available,
    is_torch_available,
    logging,
    requires_backends,
    strtobool,
    torch_only_method,
)

# 如果安装了 sacremoses 包，则导入
if is_sacremoses_available():
    import sacremoses as sm

# 如果安装了 torch 包，则导入
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义不同文件的名称
VOCAB_FILES_NAMES = {
    "pretrained_vocab_file": "vocab.pkl",
    "pretrained_vocab_file_torch": "vocab.bin",
    "vocab_file": "vocab.txt",
}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "pretrained_vocab_file": {
        "transfo-xl-wt103": "https://huggingface.co/transfo-xl-wt103/resolve/main/vocab.pkl",
    }
}

# 预训练位置嵌入长度映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "transfo-xl-wt103": None,
}

# 预训练语料归档映射
PRETRAINED_CORPUS_ARCHIVE_MAP = {
    "transfo-xl-wt103": "https://huggingface.co/transfo-xl-wt103/resolve/main/corpus.bin",
}
CORPUS_NAME = "corpus.bin"

# 匹配数字正则表达式
MATCH_NUMBERS = r"(?<=\d)[,.](?=\d)", r" @\g<0>@ "
DETOKENIZE_NUMBERS = [(r" @\,@ ", r","), (r" @\.@ ", r".")]

# 将数字拆分为单独的标记
def tokenize_numbers(text_array: List[str]) -> List[str]:
    """
    Splits large comma-separated numbers and floating point values. This is done by replacing commas with ' @,@ ' and
    dots with ' @.@ '.

    Args:
        text_array: An already tokenized text as list.

    Returns:
        A list of strings with tokenized numbers.

    Example:

    ```python
    >>> tokenize_numbers(["$", "5,000", "1.73", "m"])
    ['$', '5', '@,@', '000', '1', '@.@', '73', 'm']
    ```"""
    tokenized = []
    for i in range(len(text_array)):
        reg, sub = MATCH_NUMBERS
        replaced = re.sub(reg, sub, text_array[i]).split()
        tokenized.extend(replaced)

    return tokenized

# 将已经标记为��字的文本还原
def detokenize_numbers(text: str) -> str:
    """
    Inverts the operation of *tokenize_numbers*. This is replacing ' @,@ ' and ' @.@' by ',' and '.'.

    Args:
        text: A string where the number should be detokenized.

    Returns:
        A detokenized string.
    # 对给定的文本进行数字解标记化
    def detokenize_numbers(text):
        # 遍历解标记化的正则表达式和替换规则
        for reg, sub in DETOKENIZE_NUMBERS:
            # 使用正则表达式替换文本中匹配正则表达式的部分，替换为指定的字符串
            text = re.sub(reg, sub, text)
        # 返回解标记化后的文本
        return text
    # 定义了一个 TransfoXLTokenizer 类，该类继承自 PreTrainedTokenizer
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).
    """

    # 定义了一些类变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    # 定义了初始化方法
    def __init__(
        self,
        special=None,  # 一些特殊符号
        min_freq=0,  # 单词在词汇表中的最小出现次数
        max_size=None,  # 词汇表的最大大小
        lower_case=False,  # 是否将输入转换为小写
        delimiter=None,  # 用于分隔单词的分隔符
        vocab_file=None,  # 包含词汇表的文件
        pretrained_vocab_file: str = None,  # 包含预先训练词汇表的文件
        never_split=None,  # 永不分割的单词列表
        unk_token="<unk>",  # 未知单词标记
        eos_token="<eos>",  # 序列结束标记
        additional_special_tokens=["<formula>"],  # 附加特殊符号列表
        language="en",  # tokenizer 的语言
        **kwargs,  # 其它参数
    @property
    """
    # 返回小写标志
    def do_lower_case(self):
        return self.lower_case

    # 编译有关标点符号周围空格的模式
    def _compile_space_around_punctuation_pattern(self):
        # 创建以特殊标点符号为前瞻的正则表达式模式
        look_ahead_for_special_token = f"(?=[{self.punctuation_symbols}])"
        # 创建以除空格之外所有字符为前瞻的正则表达式模式
        look_ahead_to_match_all_except_space = r"(?=[^\s])"
        # 返回编译后的正则表达式对象
        return re.compile(r"" + look_ahead_for_special_token + look_ahead_to_match_all_except_space)

    # 统计文件中的内容
    def count_file(self, path, verbose=False, add_eos=False):
        # 如果启用详细模式，则记录日志信息
        if verbose:
            logger.info(f"counting file {path} ...")
        # 断言文件路径存在
        assert os.path.exists(path), f"Input file {path} not found"

        # 创建用于存储句子的列表
        sents = []
        with open(path, "r", encoding="utf-8") as f:
            # 遍历文件中的每一行
            for idx, line in enumerate(f):
                # 如果启用详细模式并且已经处理了一定数量的行，则记录日志信息
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                # 对每行数据进行标记化处理，将结果存储在符号列表中
                symbols = self.tokenize(line, add_eos=add_eos)
                # 更新符号计数器
                self.counter.update(symbols)
                # 将处理后的符号列表存储在句子列表中
                sents.append(symbols)

        return sents

    # 统计句子中的符号
    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        # 如果启用详细模式，则记录日志信息
        if verbose:
            logger.info(f"counting {len(sents)} sents ...")
        # 遍历句子列表中的每个句子
        for idx, symbols in enumerate(sents):
            # 如果启用详细模式并且已经处理了一定数量的句子，则记录日志信息
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            # 更新符号计数器
            self.counter.update(symbols)

    # 从文件构建对象
    def _build_from_file(self, vocab_file):
        # 初始化符号到索引和索引到符号的字典
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, "r", encoding="utf-8") as f:
            # 遍历文件中的每一行
            for line in f:
                # 提取每行的第一个符号
                symb = line.strip().split()[0]
                # 将符号添加到符号表中
                self.add_symbol(symb)
        # 检查是否存在"<UNK>"或"<unk>"符号，如果不存在则引发异常
        if "<UNK>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<UNK>"]
        elif "<unk>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<unk>"]
        else:
            raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    # 保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 构建保存词汇表文件路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["pretrained_vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 将当前对象保存到文件
        with open(vocab_file, "wb") as f:
            pickle.dump(self.__dict__, f)
        return (vocab_file,)
    # 构建词汇表
    def build_vocab(self):
        # 如果存在预定义的词汇表文件
        if self.vocab_file:
            # 打印日志，提示正在从文件构建词汇表
            logger.info(f"building vocab from {self.vocab_file}")
            # 从文件构建词汇表
            self._build_from_file(self.vocab_file)
            # 打印日志，提示最终词汇表大小
            logger.info(f"Final vocab size {len(self.sym2idx)}")
        else:
            # 如果没有预定义的词汇表文件
            # 打印日志，提示将根据最小频率和最大大小构建词汇表
            logger.info(f"building vocab with min_freq={self.min_freq}, max_size={self.max_size}")
            # 初始化索引到符号和符号到索引的字典
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            # 将特殊符号添加到词汇表中
            for sym in self.special:
                self.add_special(sym)

            # 遍历最常见的符号，并根据最小频率和最大大小添加符号到词汇表中
            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            # 打印日志，提示最终词汇表大小和原始唯一标记数
            logger.info(f"Final vocab size {len(self.sym2idx)} from {len(self.counter)} unique tokens")

    # 编码文件
    @torch_only_method
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
        # 如果需要详细输出日志信息
        if verbose:
            logger.info(f"encoding file {path} ...")
        # 检查输出文件是否存在
        assert os.path.exists(path), f"Output file {path} not found"
        encoded = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # 如果需要详细输出日志信息，并且大于指定行数则输出日志
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                # 根据指定方法将行分词并编码
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        # 如果需要有序输出，则连接编码数据
        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 编码句子列表
    @torch_only_method
    def encode_sents(self, sents, ordered=False, verbose=False):
        # 如果需要详细输出日志信息
        if verbose:
            logger.info(f"encoding {len(sents)} sents ...")
        encoded = []
        for idx, symbols in enumerate(sents):
            # 如果需要详细输出日志信息，并且大于指定行数则输出日志
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            encoded.append(self.convert_to_tensor(symbols))

        # 如果需要有序输出，则连接编码数据
        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 添加特殊符号到词汇表
    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            # 根据特殊符号生成对应的索引属性
            setattr(self, f"{sym.strip('<>')}_idx", self.sym2idx[sym])

    # 添加符号到词汇表
    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
    # 移动一个添加的 token 到词汇表中的特定位置。当调整嵌入层大小时，应该使用此方法，将 tokenizer 中的 token 从默认位置（最后）移动到所需位置。
    def move_added_token(self, token: str, target_idx: int):
        # 检查 token 是否在添加的 token 中
        assert token in self.added_tokens_encoder, "Token which should be moved has to be an added token"
        # 检查 token 是否已存在于词汇表中
        assert token not in self.idx2sym, "Token which should be moved is already in vocab"
    
        # 在词汇表中插入 token
        self.idx2sym.insert(target_idx, token)
        self.sym2idx[token] = target_idx
    
        # 调整 sym2idx 中后续的索引
        for idx in range(target_idx + 1, len(self.idx2sym)):
            current_sym = self.idx2sym[idx]
            self.sym2idx[current_sym] = idx
    
        # 从 added_tokens 中删除 token
        old_index = self._added_tokens_encoder.pop(token)
        self._added_tokens_decoder.pop(old_index)
    
    # 对文本进行 Moses 标点符号归一化
    def moses_punct_norm(self, text):
        return self.moses_punct_normalizer.normalize(text)
    
    # 使用 Moses 分词器对文本进行分词
    def moses_tokenize(self, text):
        return self.moses_tokenizer.tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split
        )
    
    # 使用 Moses 标点符号归一化和分词器进行基本分词
    def moses_pipeline(self, text: str) -> List[str]:
        text = self.moses_punct_norm(text) # 对文本进行 Moses 标点符号归一化
        text = self.moses_tokenize(text)    # 使用 Moses 分词器对文本进行分词
        text = tokenize_numbers(text)       # 对数字进行分词
        return text
    
    # 将 id 转换成 token（BPE）使用词汇表
    def _convert_id_to_token(self, idx):
        # 检查索引是否在词汇表范围内
        assert 0 <= idx < len(self), f"Index {idx} out of vocabulary range"
        return self.idx2sym[idx]
    def _convert_token_to_id(self, sym):
        """将词符（字符串）转换为词汇表中的 id"""
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # 如果词符不在词汇表中
            if hasattr(self, "unk_idx"):
                return self.sym2idx.get(sym, self.unk_idx)
            # 与预训练模型的向后兼容性
            elif "<unk>" in self.sym2idx:
                return self.sym2idx["<unk>"]
            elif "<UNK>" in self.sym2idx:
                return self.sym2idx["<UNK>"]
            else:
                raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    def convert_tokens_to_string(self, tokens):
        """将一系列词符（字符串）转换为单个字符串。此外，拆分的数字会转回到其原始形式。"""
        out_string = self.moses_detokenizer.detokenize(tokens)
        return detokenize_numbers(out_string).strip()

    @torch_only_method
    def convert_to_tensor(self, symbols):
        """将符号转换为张量"""
        return torch.LongTensor(self.convert_tokens_to_ids(symbols))

    @property
    def vocab_size(self):
        return len(self.idx2sym)

    def get_vocab(self):
        """获取词汇表"""
        vocab = self.sym2idx.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, line, add_eos=False, add_double_eos=False):
        """对行进行标记化，可以选择添加 eos 标记和 double eos 标记"""
        line = line.strip()
        # 转换为小写
        if self.lower_case:
            line = line.lower()

        # 如果分隔符为空，则词符为整行
        if self.delimiter == "":
            symbols = line
        else:
            symbols = self.moses_pipeline(line)

        if add_double_eos:  # 对 lm1b 数据集专用，添加两个开始符号
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols
class LMOrderedIterator(object):
    # 初始化函数，设置参数和属性
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        # 设定批量大小
        self.bsz = bsz
        # 设定时间步长
        self.bptt = bptt
        # 设定外部长度
        self.ext_len = ext_len if ext_len is not None else 0
        # 设定设备
        self.device = device

        # 计算数据集可以被批量大小整除的步数
        self.n_step = data.size(0) // bsz

        # 剪切掉多余的元素，使其可以整除
        data = data.narrow(0, 0, self.n_step * bsz)

        # 将数据均匀分配到每个批次中
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # 计算迭代的次数
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    # 获取单个批次的数据
    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        data_out = data.transpose(0, 1).contiguous().to(self.device)
        target_out = target.transpose(0, 1).contiguous().to(self.device)

        return data_out, target_out, seq_len

    # 获取固定长度的迭代器
    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    # 获取可变长度的迭代器
    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    # 迭代器方法
    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    # 初始化函数，设置参数和属性
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        # 设置数据
        self.data = data
        # 设置批量大小
        self.bsz = bsz
        # 设置时间步长
        self.bptt = bptt
        # 设置外部长度
        self.ext_len = ext_len if ext_len is not None else 0
        # 设置设备
        self.device = device
        # 是否打乱数据
        self.shuffle = shuffle

    # 获取句子流
    def get_sent_stream(self):
        # 获取索引迭代器
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle else np.array(range(len(self.data)))

        # 获取句子迭代器
        for idx in epoch_indices:
            yield self.data[idx]

    # 仅限于torch的方法
    @torch_only_method
    # 定义一个流迭代器，接收一个数据流（sent_stream）
    def stream_iterator(self, sent_stream):
        # 为每个批次中的数据创建流对象，初始化为None
        streams = [None] * self.bsz
    
        # 创建大小为 self.bptt x self.bsz 的 LongTensor 对象 data 和 target
        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)
    
        # 记录要保留的旧数据条目数，初始化为 0
        n_retain = 0
    
        # 无限循环，直到函数内部使用 return 关键字结束循环
        while True:
            # 清空 data 的旧数据，并填充为 -1
            data[n_retain:].fill_(-1)
            # 用 -1 填充 target
            target.fill_(-1)
    
            # 初始化标识符 valid_batch 为 True
            valid_batch = True
    
            # 遍历批次中的每个元素
            for i in range(self.bsz):
                # 初始化已填充数据的数量为 0
                n_filled = 0
                try:
                    # 当还没有填充满当前批次的数据时执行循环
                    while n_filled < self.bptt:
                        # 如果 streams[i] 为空或者长度小于等于 1，从 sent_stream 中获取下一个流对象
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # 计算本次需要填充的数据量
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # 将前 n_retain 个仍需要保留的数据从上一批次拷贝过来
                        data[n_retain + n_filled : n_retain + n_filled + n_new, i] = streams[i][:n_new]
                        # 填充 target 的数据
                        target[n_filled : n_filled + n_new, i] = streams[i][1 : n_new + 1]
                        # 更新流对象，丢弃已经填充到数据里的 token
                        streams[i] = streams[i][n_new:]
                        # 更新已填充的数据数量
                        n_filled += n_new
                except StopIteration:
                    # 当 sent_stream 已经迭代完毕，设置 valid_batch 标志为 False，跳出循环
                    valid_batch = False
                    break
    
            # 如果 valid_batch 为 False，直接返回
            if not valid_batch:
                return
    
            # 调整数据的维度，转置并移动到设备上
            data_out = data.transpose(0, 1).contiguous().to(self.device)
            target_out = target.transpose(0, 1).contiguous().to(self.device)
    
            # 生成数据和目标序列数据的元组，并返回
            yield data_out, target_out, self.bptt
    
            # 更新 n_retain 的值为数据的长度或者 ext_len 中的较小值
            n_retain = min(data.size(0), self.ext_len)
            # 如果 n_retain 大于 0，则将后面的数据拷贝到前面，保留旧数据
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            # 重新调整 data 的大小
            data.resize_(n_retain + self.bptt, data.size(1))
    
    # 重写 __iter__ 方法
    def __iter__(self):
        # 获取 sent_stream 的迭代器
        sent_stream = self.get_sent_stream()
    
        # 遍历 stream_iterator 生成的每个批次数据，并通过 yield 关键字返回
        for batch in self.stream_iterator(sent_stream):
            yield batch
class LMMultiFileIterator(LMShuffledIterator):
    # LMMultiFileIterator 类的初始化方法
    def __init__(self, paths, vocab, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        # 初始化文件路径列表
        self.paths = paths
        # 初始化词汇表
        self.vocab = vocab
        # 初始化批大小
        self.bsz = bsz
        # 初始化 BPTT（backpropagation through time）长度
        self.bptt = bptt
        # 初始化扩展长度，默认为0
        self.ext_len = ext_len if ext_len is not None else 0
        # 初始化设备，默认为 CPU
        self.device = device
        # 是否对文件进行洗牌
        self.shuffle = shuffle

    # 获取文件句子流的方法
    def get_sent_stream(self, path):
        # 使用词汇表编码文件，添加双端标记，并返回句子列表
        sents = self.vocab.encode_file(path, add_double_eos=True)
        # 如果需要洗牌
        if self.shuffle:
            # 随机打乱句子列表顺序
            np.random.shuffle(sents)
        # 将句子列表转换为迭代器
        sent_stream = iter(sents)

        # 返回句子流迭代器
        return sent_stream

    # 实现迭代器协议的方法
    def __iter__(self):
        # 如果需要洗牌
        if self.shuffle:
            # 随机打乱文件路径列表顺序
            np.random.shuffle(self.paths)

        # 遍历文件路径列表
        for path in self.paths:
            # 获取文件句子流迭代器
            sent_stream = self.get_sent_stream(path)
            # 调用 stream_iterator 方法生成批次
            for batch in self.stream_iterator(sent_stream):
                # 生成一个批次并 yield 返回
                yield batch


class TransfoXLCorpus(object):
    # 从预训练模型加载语料库的类方法
    @classmethod
    @torch_only_method
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a pre-processed corpus.
        """
        # 从预训练模型名称或路径加载词汇表
        vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # 判断是否是本地文件
        is_local = os.path.isdir(pretrained_model_name_or_path)
        # 尝试从缓存中获取语料文件路径
        try:
            resolved_corpus_file = cached_file(pretrained_model_name_or_path, CORPUS_NAME, cache_dir=cache_dir)
        except EnvironmentError:
            # 如果语料文件未找到，记录错误信息
            logger.error(
                f"Corpus '{pretrained_model_name_or_path}' was not found in corpus list"
                f" ({', '.join(PRETRAINED_CORPUS_ARCHIVE_MAP.keys())}. We assumed '{pretrained_model_name_or_path}'"
                f" was a path or url but couldn't find files {CORPUS_NAME} at this path or url."
            )
            # 返回 None
            return None
        # 如果是本地文件，记录加载过程信息
        if is_local:
            logger.info(f"loading corpus file {resolved_corpus_file}")
        else:
            logger.info(f"loading corpus file {CORPUS_NAME} from cache at {resolved_corpus_file}")

        # 实例化 TransfoXLCorpus 对象
        corpus = cls(*inputs, **kwargs)
        # 从文件中加载语料库字典
        corpus_dict = torch.load(resolved_corpus_file)
        # 将语料库字典的键值对添加到 corpus 对象的属性中
        for key, value in corpus_dict.items():
            corpus.__dict__[key] = value
        # 将词汇表添加到 corpus 对象的属性中
        corpus.vocab = vocab
        # 将语料库中的训练、验证和测试数据转换为 Tensor 类型
        if corpus.train is not None:
            corpus.train = torch.tensor(corpus.train, dtype=torch.long)
        if corpus.valid is not None:
            corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)
        if corpus.test is not None:
            corpus.test = torch.tensor(corpus.test, dtype=torch.long)
        # 返回 corpus 对象
        return corpus

    # TransfoXLCorpus 类的初始化方法
    def __init__(self, *args, **kwargs):
        # 初始化 TransfoXLTokenizer 对象
        self.vocab = TransfoXLTokenizer(*args, **kwargs)
        # 初始化数据集
        self.dataset = None
        # 初始化训练数据
        self.train = None
        # 初始化验证数据
        self.valid = None
        # 初始化测试数据
        self.test = None
```  
    # 构建语料库的函数，根据给定的路径和数据集名称来构建
    def build_corpus(self, path, dataset):
        # 将数据集名称赋值给对象的 dataset 属性
        self.dataset = dataset

        # 根据不同数据集，统计相应文件的词频
        if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        # 对于 "wt103" 数据集，只统计训练文件的词频
        elif self.dataset == "wt103":
            self.vocab.count_file(os.path.join(path, "train.txt"))
        # 对于 "lm1b" 数据集，则根据文件路径模式匹配训练文件
            train_paths = glob.glob(train_path_pattern)
            # 在执行 build_vocab() 时，vocab 对象会根据文件加载词汇表

        # 构建词汇表
        self.vocab.build_vocab()

        # 根据不同数据集，将文件编码成整数序列
        if self.dataset in ["ptb", "wt2", "wt103"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True)
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True, add_eos=False)
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=False, add_double_eos=True)

    # 获取数据迭代器的函数，根据数据集名称和划分方式来选择对应的迭代器
    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                # 对于训练集，使用有序迭代器
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == "lm1b":
                kwargs["shuffle"] = True
                # 对于 "lm1b" 数据集，使用多文件迭代器
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                # 对于验证集和测试集，使用有序迭代器
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == "lm1b":
                # 对于 "lm1b" 数据集，使用打乱顺序迭代器
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        else:
            data_iter = None
            # 如果划分方式不被识别，则引发异常
            raise ValueError(f"Split not recognized: {split}")

        return data_iter
# 限定此函数仅能在 Torch 环境下使用
@torch_only_method
# 获取语言模型的语料库
def get_lm_corpus(datadir, dataset):
    # 构建缓存文件路径
    fn = os.path.join(datadir, "cache.pt")
    # 构建 pickle 文件的缓存文件路径
    fn_pickle = os.path.join(datadir, "cache.pkl")
    # 如果缓存文件存在
    if os.path.exists(fn):
        # 打印日志，提示正在加载缓存的数据集
        logger.info("Loading cached dataset...")
        # 加载语料库数据集
        corpus = torch.load(fn_pickle)
    # 如果 pickle 文件的缓存文件存在
    elif os.path.exists(fn):
        # 打印日志，提示正在从 pickle 文件中加载缓存的数据集
        logger.info("Loading cached dataset from pickle...")
        # 如果环境变量 TRUST_REMOTE_CODE 未设置为 True，则抛出 ValueError 异常
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        # 以二进制读模式打开缓存文件，并加载语料库数据集
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    # 如果缓存文件不存在
    else:
        # 打印日志，提示正在生成指定数据集的数据
        logger.info(f"Producing dataset {dataset}...")
        # 初始化额外参数字典
        kwargs = {}
        # 如果数据集为 "wt103" 或 "wt2"
        if dataset in ["wt103", "wt2"]:
            # 添加特殊标记 "<eos>"
            kwargs["special"] = ["<eos>"]
            # 设置字母大小写不转换
            kwargs["lower_case"] = False
        # 如果数据集为 "ptb"
        elif dataset == "ptb":
            # 添加特殊标记 "<eos>"
            kwargs["special"] = ["<eos>"]
            # 设置字母小写
            kwargs["lower_case"] = True
        # 如果数据集为 "lm1b"
        elif dataset == "lm1b":
            # 不添加特殊标记
            kwargs["special"] = []
            # 设置字母大小写不转换
            kwargs["lower_case"] = False
            # 设置词汇表文件路径
            kwargs["vocab_file"] = os.path.join(datadir, "1b_word_vocab.txt")
        # 如果数据集为 "enwik8" 或 "text8"，则无需额外处理，直接跳过
        elif dataset in ["enwik8", "text8"]:
            pass

        # 根据给定的数据集名和参数初始化 TransfoXLCorpus 对象
        corpus = TransfoXLCorpus(datadir, dataset, **kwargs)
        # 将语料库数据集保存到缓存文件中
        torch.save(corpus, fn)

    # 返回语料库数据集
    return corpus
```