# `.\transformers\data\datasets\language_modeling.py`

```
# 导入所需的模块和库
import json  # 导入用于 JSON 数据处理的模块
import os  # 导入用于操作操作系统功能的模块
import pickle  # 导入用于序列化和反序列化 Python 对象的模块
import random  # 导入用于生成随机数的模块
import time  # 导入用于时间相关操作的模块
import warnings  # 导入用于警告处理的模块
from typing import Dict, List, Optional  # 导入用于类型提示的模块

import torch  # 导入 PyTorch 库
from filelock import FileLock  # 导入用于文件锁的模块
from torch.utils.data import Dataset  # 导入 PyTorch 数据集类

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器类
from ...utils import logging  # 导入日志记录模块

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 弃用警告消息
DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

# 文本数据集类
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,  # 预训练分词器
        file_path: str,  # 文件路径
        block_size: int,  # 数据块大小
        overwrite_cache=False,  # 是否覆盖缓存
        cache_dir: Optional[str] = None,  # 缓存目录，默认为None
    ):
        # 发出即将弃用的警告，提供相关链接以获取更多信息
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 如果文件路径不存在，引发数值错误
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        # 减去配对模式下特殊标记的数量以计算块大小
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        # 拆分文件路径，获取目录和文件名
        directory, filename = os.path.split(file_path)
        # 创建缓存特征文件路径
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # 确保在分布式训练中只有第一个进程处理数据集，其他进程使用缓存
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            # 如果缓存特征文件存在且不覆盖缓存，则加载缓存数据
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                # 否则，从数据集文件中创建特征
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                # 使用分词器将文本转换为标记的 ID
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                # 按块大小截断文本
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # 注意：为简化起见（无填充），这里丢弃了最后一个截断的示例
                # 如果你的数据集较小，首先你应该寻找一个更大的数据集 :-) 其次，
                # 你可以通过添加（特定于模型的）填充来改变这种行为。

                start = time.time()
                # 将特征保存到缓存文件中
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        # 返回示例列表的长度
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        # 返回指定索引处的示例，并将其转换为 Torch 张量
        return torch.tensor(self.examples[i], dtype=torch.long)
# 创建一个用于逐行读取文本数据的数据集类，准备逐行读取文本数据的数据集类以后会被一个与框架无关的方法取代
class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    # 初始化方法，接受分词器、文件路径和块大小作为参数
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        # 发出警告，提示此方法即将被一个与框架无关的方法取代，同时提供了一个链接以供查看详细信息
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 如果指定路径的文件不存在，抛出值错误异常
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # 不缓存特征，基于以下假设：我们很快将在所有地方使用`tokenizers`仓库中的快速多线程分词器
        logger.info(f"Creating features from dataset file at {file_path}")
        
        # 使用UTF-8编码打开指定路径的文件，并读取文件中的非空行
        with open(file_path, encoding="utf-8") as f:
            # 将非空行添加到列表中，去除了行尾的换行符
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        # 使用分词器对读取的文本行进行分词，添加特殊标记并截断到指定的块大小
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        # 将分词后的输入 ID 存储为示例，作为数据集的一部分
        self.examples = batch_encoding["input_ids"]
        # 将示例转换为字典格式，键为"input_ids"，值为对应的张量
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    # 返回数据集中示例的数量
    def __len__(self):
        return len(self.examples)

    # 根据索引返回对应的示例，以字典格式包含键"input_ids"和对应的张量值
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


# 定义另一个数据集类，用于逐行读取具有参考数据的文本
class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    # 初始化函数，接受一个分词器对象、文件路径、块大小和参考文件路径作为参数
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        # 发出关于即将弃用的警告，包含一个链接指向示例代码的地址
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        # 如果文件路径不存在，则抛出 ValueError 异常
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # 如果参考文件路径不存在，则抛出 ValueError 异常
        if os.path.isfile(ref_path) is False:
            raise ValueError(f"Ref file path {file_path} not found")
        # 记录日志，指示正在从数据集文件创建特征
        logger.info(f"Creating features from dataset file at {file_path}")
        # 记录日志，指示正在使用参考段落结果
        logger.info(f"Use ref segment results at {ref_path}")
        # 打开数据集文件，读取所有行
        with open(file_path, encoding="utf-8") as f:
            # 使用 readlines() 方法读取文件的所有行，并将其存储在列表中
            data = f.readlines()  # use this method to avoid delimiter '\u2029' to split a line
        # 清理数据：去除每行两边的空白字符，并且确保行不为空
        data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]
        # 从文件中获取参考信息
        with open(ref_path, encoding="utf-8") as f:
            # 使用 read().splitlines() 方法读取文件的所有行，并将其拆分成行的列表
            ref = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        # 检查数据和参考的长度是否一致，如果不一致则抛出 ValueError 异常
        if len(data) != len(ref):
            raise ValueError(
                f"Length of Input file should be equal to Ref file. But the length of {file_path} is {len(data)} "
                f"while length of {ref_path} is {len(ref)}"
            )

        # 使用分词器对数据进行编码，添加特殊标记、截断和限制最大长度，并将结果存储在 batch_encoding 中
        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        # 将编码后的数据存储在 self.examples 中，每个样本使用字典存储
        self.examples = batch_encoding["input_ids"]
        # 将每个样本转换为字典格式，并存储在 self.examples 中
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        # 计算样本数量
        n = len(self.examples)
        # 为每个样本添加对应的参考信息
        for i in range(n):
            # 使用 torch.tensor 创建参考信息的张量，并将其存储在 self.examples[i]["chinese_ref"] 中
            self.examples[i]["chinese_ref"] = torch.tensor(ref[i], dtype=torch.long)

    # 获取数据集的长度
    def __len__(self):
        return len(self.examples)

    # 获取指定索引的样本
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        # 发出警告，提醒使用者此功能即将被弃用
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 检查文件目录是否存在
        if os.path.isdir(file_dir) is False:
            raise ValueError(f"{file_dir} is not a directory")
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        # TODO: randomness could apply a random seed, ex. rng = random.Random(random_seed)
        # 文件路径类似于 ./dataset/wiki_1, ./dataset/wiki_2
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            # 检查文件是否存在
            if os.path.isfile(file_path) is False:
                raise ValueError(f"{file_path} is not a file")
            article_open = False
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if "<doc id=" in line:
                        article_open = True
                    elif "</doc>" in line:
                        article_open = False
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)

        logger.info("Dataset parse finished.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        short_seq_probability=0.1,
        nsp_probability=0.5,
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
```