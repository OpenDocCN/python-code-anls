# `.\lucidrains\electra-pytorch\pretraining\openwebtext\dataset.py`

```py
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from functools import partial
from pathlib import Path

import numpy as np

import torch
import torch.utils.data

from openwebtext import tokenization


class ExampleBuilder:
    """Given a stream of input text, creates pretraining examples."""

    def __init__(self, vocab, max_length):
        # 初始化 ExampleBuilder 类，传入词汇表和最大长度参数
        self._vocab = vocab
        self._current_sentences = []  # 当前正在构建的例子的句子列表
        self._current_length = 0  # 当前正在构建的例子的长度
        self._max_length = max_length  # 最大长度
        self._target_length = max_length  # 目标长度

    def add_line(self, bert_tokids):
        """Adds a line of text to the current example being built."""
        # 将一行文本添加到当前正在构建的例子中
        self._current_sentences.append(bert_tokids)  # 将 BERT token ids 添加到当前句子列表中
        self._current_length += len(bert_tokids)  # 更新当前例子的长度
        if self._current_length >= self._target_length:
            return self._create_example()  # 如果当前长度达到目标长度，则创建一个例子
        return None

    def _create_example(self):
        """Creates a pre-training example from the current list of sentences."""
        # 有很小的概率只有一个段落，类似分类任务
        if random.random() < 0.1:
            first_segment_target_length = 100000
        else:
            # -3 是因为输入文本中尚未有 [CLS]/[SEP] 标记
            first_segment_target_length = (self._target_length - 3) // 2

        first_segment = []  # 第一个段落
        second_segment = []  # 第二个段落
        for sentence in self._current_sentences:
            # 如果第一个段落为空，或者加入当前句子不会超过目标长度，或者50%的概率加入当前句子会超过目标长度但第二个段落为空
            if (len(first_segment) == 0 or
                len(first_segment) + len(sentence) < first_segment_target_length or
                (len(second_segment) == 0 and
                len(first_segment) < first_segment_target_length and
                random.random() < 0.5)):
                first_segment += sentence  # 将当前句子加入第一个段落
            else:
                second_segment += sentence  # 将当前句子加入第二个段落

        # 裁剪到最大长度，考虑尚未添加的 [CLS]/[SEP] 标记
        first_segment = first_segment[:self._max_length - 2]
        second_segment = second_segment[:max(0, self._max_length - len(first_segment) - 3)]

        # 准备开始构建下一个例子
        self._current_sentences = []  # 清空当前句子列表
        self._current_length = 0  # 重置当前长度
        # 有很小的概率选择随机长度而不是最大长度
        if random.random() < 0.05:
            self._target_length = random.randint(5, self._max_length)
        else:
            self._target_length = self._max_length

        return self._make_tf_example(first_segment, second_segment)  # 创建 TF 格式的例子
    def _make_tf_example(self, first_segment, second_segment):
        """将两个文本“段”转换为tf.train.Example。"""
        # 获取词汇表
        vocab = self._vocab
        # 构建输入文本的token id序列，包括[CLS]和[SEP]标记
        input_ids = [vocab["[CLS]"]] + first_segment + [vocab["[SEP]"]
        # 初始化段落标识符，全部为0
        segment_ids = [0] * len(input_ids)
        # 如果存在第二个文本段
        if second_segment:
            # 添加第二个文本段的token id序列和段落标识符
            input_ids += second_segment + [vocab["[SEP]"]]
            segment_ids += [1] * (len(second_segment) + 1)
        # 初始化输入掩码，全部为1
        input_mask = [1] * len(input_ids)
        # 将输入文本的token id序列、输入掩码和段落标识符填充至最大长度
        input_ids += [0] * (self._max_length - len(input_ids))
        input_mask += [0] * (self._max_length - len(input_mask))
        segment_ids += [0] * (self._max_length - len(segment_ids)

        # 定义创建整数特征的函数
        def create_int_feature(tensors):
            return torch.tensor(tensors)

        # 构建tf.train.Example对象
        tf_example = {
            "input_ids": create_int_feature(input_ids),
            "input_mask": create_int_feature(input_mask),
            "segment_ids": create_int_feature(segment_ids)
        }
        return tf_example
# 定义一个继承自torch.utils.data.IterableDataset的OpenWebTextDataset类
class OpenWebTextDataset(torch.utils.data.IterableDataset):
    # 初始化方法，接收feature_set_paths和n_tensors_per_file两个参数
    def __init__(self, feature_set_paths, n_tensors_per_file):
        # 将feature_set_paths赋值给实例变量feature_set_paths
        self.feature_set_paths = feature_set_paths
        # 将n_tensors_per_file赋值给实例变量n_tensors_per_file

    # 静态方法，用于解析文件，接收file_index作为参数
    @staticmethod
    def parse_file(file_index):
        # 尝试加载文件内容为features
        try:
            features = torch.load(str(file_index))
            # 生成器，逐个返回features中的元素
            yield from features
        # 捕获RuntimeError异常
        except RuntimeError:
            # 抛出带有文件索引信息的RuntimeError异常
            raise RuntimeError(f'Corrupted file {file_index}')

    # 返回数据集的长度
    def __len__(self):
        return len(self.feature_set_paths) * self.n_tensors_per_file

    # 迭代器方法，返回一个可迭代对象
    def __iter__(self):
        # 使用map函数将parse_file应用于feature_set_paths中的每个元素，然后使用chain.from_iterable将结果展平
        return chain.from_iterable(map(self.parse_file, self.feature_set_paths))


# 定义一个继承自torch.utils.data.IterableDataset的ExampleBuilderDataset类
class ExampleBuilderDataset(torch.utils.data.IterableDataset):
    # 初始化方法，接收dataset和builder两个参数
    def __init__(self, dataset, builder):
        # 将dataset赋值给实例变量dataset
        self.dataset = dataset
        # 将builder赋值给实例变量builder

    # 返回数据集的长度
    def __len__(self):
        return len(self.dataset)

    # 迭代器方法，返回一个可迭代对象
    def __iter__(self):
        # 定义一个内部函数create_example
        def create_example():
            # 无限循环
            while True:
                # 获取下一个dataset元素，转换为CPU上的numpy数组，然后转换为列表
                token_ids = list(next(self.dataset).cpu().numpy())
                # 使用builder的add_line方法添加token_ids，如果返回了example，则返回该example
                example = self.builder.add_line(token_ids)
                if example:
                    return example

        # 无限循环
        while True:
            # 生成器，逐个返回create_example函数的结果
            yield create_example()


# 定义一个循环生成器函数cycle
def cycle(iterable):
    # 无限循环
    while True:
        # 遍历可迭代对象iterable，逐个返回元素
        for x in iterable:
            yield x


# 定义一个函数new_tokenizer，接收vocab_file和do_lower_case两个参数
def new_tokenizer(vocab_file, do_lower_case=True):
    # 返回一个FullTokenizer对象，传入vocab_file和do_lower_case参数
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


# 定义一个函数parse_tokenizer，接收tokenizer和text两个参数
def parse_tokenizer(tokenizer, text):
    # 将text转换为token ids并返回
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


# 定义一个函数create_tokenizer，接收vocab_file和do_lower_case两个参数
def create_tokenizer(vocab_file, do_lower_case=True):
    # 创建一个FullTokenizer对象
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    # 返回一个partial对象，传入parse_tokenizer函数和tokenizer参数
    return partial(parse_tokenizer, tokenizer)


# 定义一个函数load_owt，接收owt_dir和n_tensors_per_file两个参数
def load_owt(owt_dir, n_tensors_per_file):
    # 将owt_dir转换为Path对象
    owt_dir_path = Path(owt_dir)
    # 获取owt_dir_path目录下的所有文件路径，随机打乱顺序
    feature_set_paths = [owt_dir_path / feature_set_path for feature_set_path in os.listdir(owt_dir_path)]
    np.random.shuffle(feature_set_paths)
    # 断言feature_set_paths长度大于0
    assert len(feature_set_paths) > 0
    # 返回一个OpenWebTextDataset对象，传入feature_set_paths和n_tensors_per_file参数
    return OpenWebTextDataset(feature_set_paths, n_tensors_per_file=n_tensors_per_file)


# 定义一个函数wrap_example_builder，接收dataset、vocab和max_length三个参数
def wrap_example_builder(dataset, vocab, max_length):
    # 返回一个ExampleBuilderDataset对象，传入循环生成器cycle(iter(dataset))和ExampleBuilder(vocab, max_length)参数
    return ExampleBuilderDataset(cycle(iter(dataset)), ExampleBuilder(vocab, max_length))
```