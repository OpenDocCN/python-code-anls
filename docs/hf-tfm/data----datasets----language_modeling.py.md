# `.\data\datasets\language_modeling.py`

```py
# 导入必要的模块和库
import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from filelock import FileLock
from torch.utils.data import Dataset

# 导入相对路径的模块和库
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 弃用警告信息
DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

class TextDataset(Dataset):
    """
    这个类将很快被一个与框架无关的方法所取代。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 检查输入的文件路径是否存在，如果不存在则抛出异常
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        # 根据tokenizer的特殊token数目，调整block_size的大小
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        # 将文件路径拆分为目录和文件名
        directory, filename = os.path.split(file_path)
        # 设置缓存文件路径，包含模型名称、block_size和文件名等信息
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # 确保只有分布式训练中的第一个进程处理数据集，其他进程使用缓存
        lock_path = cached_features_file + ".lock"
        # 使用文件锁定确保并发安全性
        with FileLock(lock_path):
            # 如果缓存文件已存在且不需要覆盖，则加载缓存中的特征
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                # 初始化self.examples为空列表
                self.examples = []
                # 打开文件并读取文本内容
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                # 使用tokenizer将文本分词并转换为对应的token IDs
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                # 根据block_size将tokenized_text分割成片段，并构建特征列表self.examples
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # 注意，这里为简化起见，最后一个被截断的示例被丢弃了（没有进行填充）
                # 如果你的数据集很小，首先应该寻找更大的数据集，并且你可以通过添加（特定于模型的）填充来更改此行为。

                start = time.time()
                # 将self.examples保存到缓存文件中
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    # 返回self.examples的长度作为数据集的长度
    def __len__(self):
        return len(self.examples)

    # 根据索引返回对应的torch.Tensor对象，包含在self.examples中的数据
    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        # 发出警告，指出此方法即将被不依赖框架的方法取代
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 检查文件路径是否存在，如果不存在则引发值错误异常
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # 记录消息到日志，指示正在从文件路径创建数据集特征
        logger.info(f"Creating features from dataset file at {file_path}")

        # 使用 utf-8 编码打开文件，读取所有非空行并去除首尾空格
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        # 使用给定的分词器对行进行编码，添加特殊标记并截断到指定长度
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        # 将编码后的输入 IDs 存储在示例中
        self.examples = batch_encoding["input_ids"]
        # 将每个示例封装为包含输入 IDs 的字典，并使用长整型张量进行存储
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        # 返回示例列表的长度，即数据集中示例的数量
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        # 返回索引为 i 的示例，该示例是包含输入 IDs 的字典
        return self.examples[i]


class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        # 发出警告，指示代码的某些功能将来会被弃用，并提供了更多信息的链接
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        # 检查输入文件是否存在，如果不存在则引发 ValueError 异常
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # 检查参考文件是否存在，如果不存在则引发 ValueError 异常
        if os.path.isfile(ref_path) is False:
            raise ValueError(f"Ref file path {file_path} not found")
        
        # 不缓存特征，假设很快将在所有地方使用来自 `tokenizers` 仓库的快速多线程分词器
        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        
        # 使用 UTF-8 编码打开数据文件，并读取所有行到变量 data 中
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()  # 使用这种方法避免使用分隔符 '\u2029' 来分割行
        
        # 去除每行两端的空白字符，并排除空行，生成最终的数据列表
        data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]
        
        # 使用 UTF-8 编码打开参考文件，并按行解析每行为 JSON 对象，生成 ref 列表
        with open(ref_path, encoding="utf-8") as f:
            ref = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        # 检查数据列表和参考列表的长度是否一致，如果不一致则引发 ValueError 异常
        if len(data) != len(ref):
            raise ValueError(
                f"Length of Input file should be equal to Ref file. But the length of {file_path} is {len(data)} "
                f"while length of {ref_path} is {len(ref)}"
            )

        # 使用 tokenizer 对数据进行编码处理，添加特殊标记并截断到指定的 block_size
        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        
        # 将每个编码后的示例的 "input_ids" 存储为列表的形式，存储在 self.examples 中
        self.examples = batch_encoding["input_ids"]
        
        # 将每个 "input_ids" 转换为包含 torch.tensor 的字典形式，存储在 self.examples 中
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        # 为每个示例添加 "chinese_ref" 字段，值为参考数据的 torch.tensor 形式
        n = len(self.examples)
        for i in range(n):
            self.examples[i]["chinese_ref"] = torch.tensor(ref[i], dtype=torch.long)

    def __len__(self):
        # 返回示例列表的长度，用于确定数据集的大小
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        # 根据索引 i 返回对应的示例，为字典形式，包含 "input_ids" 和 "chinese_ref"
        return self.examples[i]
class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        # 发出警告，提醒此功能即将被弃用，并提供相关链接
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        # 如果提供的文件目录不是一个目录，则引发值错误异常
        if os.path.isdir(file_dir) is False:
            raise ValueError(f"{file_dir} is not a directory")
        # 记录信息，指出正在从指定文件夹创建数据集特征
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        # 初始化空的示例列表
        self.examples = []
        # 遍历文件目录下的每个文件名
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            # 如果文件路径不是一个文件，则引发值错误异常
            if os.path.isfile(file_path) is False:
                raise ValueError(f"{file_path} is not a file")
            # 初始化文章打开标志为假
            article_open = False
            # 打开文件，使用UTF-8编码
            with open(file_path, encoding="utf-8") as f:
                # 读取原始行
                original_lines = f.readlines()
                # 初始化文章行列表
                article_lines = []
                # 遍历原始行
                for line in original_lines:
                    # 如果当前行包含"<doc id="，表示文章开始
                    if "<doc id=" in line:
                        article_open = True
                    # 如果当前行包含"</doc>"，表示文章结束
                    elif "</doc>" in line:
                        article_open = False
                        # 将文章行列表中第二行开始（排除第一行标题）的每一行转换为token IDs
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]
                        # 根据文档创建示例，将其扩展到self.examples列表中
                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        # 清空文章行列表
                        article_lines = []
                    else:
                        # 如果文章正在打开，则将当前行添加到文章行列表中
                        if article_open:
                            article_lines.append(line)

        # 记录信息，指出数据集解析完成
        logger.info("Dataset parse finished.")

    def __len__(self):
        # 返回示例列表的长度
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        # 返回指定索引处的示例
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
    ):
        # 初始化示例列表为空
        self.examples = []

    def __len__(self):
        # 返回示例列表的长度
        return len(self.examples)

    def __getitem__(self, i):
        # 返回指定索引处的示例
        return self.examples[i]
```