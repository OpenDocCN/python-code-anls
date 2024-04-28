# `.\transformers\data\processors\utils.py`

```
# 导入必要的模块和库
import csv  # 导入用于 CSV 文件操作的模块
import dataclasses  # 导入用于定义数据类的模块
import json  # 导入用于 JSON 数据操作的模块
from dataclasses import dataclass  # 从 dataclasses 模块中导入 dataclass 装饰器
from typing import List, Optional, Union  # 导入用于类型提示的模块

from ...utils import is_tf_available, is_torch_available, logging  # 导入自定义的模块和函数

logger = logging.get_logger(__name__)  # 获取日志记录器


@dataclass  # 使用 dataclass 装饰器定义数据类
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str  # 唯一的示例标识符
    text_a: str  # 第一个序列的未标记文本
    text_b: Optional[str] = None  # 第二个序列的未标记文本（可选）
    label: Optional[str] = None  # 示例的标签（可选）

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"  # 将实例序列化为 JSON 字符串并返回


@dataclass(frozen=True)  # 使用 dataclass 装饰器定义数据类，使其不可变
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]  # 输入序列标记在词汇表中的索引
    attention_mask: Optional[List[int]] = None  # 避免对填充标记进行注意力的掩码
    token_type_ids: Optional[List[int]] = None  # 表示输入的第一部分和第二部分的段标记索引（可选）
    label: Optional[Union[int, float]] = None  # 输入对应的标签（可选），用于分类问题为整数，用于回归问题为浮点数

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"  # 将实例序列化为 JSON 字符串并返回


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""  # 用于序列分类数据集的数据转换器的基类
    def get_example_from_tensor_dict(self, tensor_dict):
        """
        从包含 TensorFlow 张量的字典中获取一个示例。

        Args:
            tensor_dict: 键和值应与相应的 GLUE tensorflow_dataset 示例匹配。
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """获取训练集的 [`InputExample`] 集合。"""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """获取开发集的 [`InputExample`] 集合。"""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """获取测试集的 [`InputExample`] 集合。"""
        raise NotImplementedError()

    def get_labels(self):
        """获取此数据集的标签列表。"""
        raise NotImplementedError()

    def tfds_map(self, example):
        """
        一些 tensorflow_datasets 数据集的格式与 GLUE 数据集的格式不同。此方法将示例转换为正确的格式。
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """读取一个以制表符分隔的值文件。"""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
class SingleSentenceClassificationProcessor(DataProcessor):
    """Generic processor for a single sentence classification data set."""

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        # 初始化函数，设置标签、示例、模式和详细信息
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        # 返回示例的数量
        return len(self.examples)

    def __getitem__(self, idx):
        # 获取指定索引的示例
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    @classmethod
    def create_from_csv(
        cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        # 从CSV文件创建处理器实例
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        # 从示例创建处理器实例
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        # 从CSV文件中添加示例
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        for i, line in enumerate(lines):
            texts.append(line[column_text])
            labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = f"{split_name}-{i}" if split_name else str(i)
                ids.append(guid)

        return self.add_examples(
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
        self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
        # 添加示例到处理器中
        ):
        # 检查标签是否与文本长度匹配
        if labels is not None and len(texts_or_text_and_labels) != len(labels):
            raise ValueError(
                f"Text and labels have mismatched lengths {len(texts_or_text_and_labels)} and {len(labels)}"
            )
        # 检查 ID 是否与文本长度匹配
        if ids is not None and len(texts_or_text_and_labels) != len(ids):
            raise ValueError(f"Text and ids have mismatched lengths {len(texts_or_text_and_labels)} and {len(ids)}")
        # 如果没有提供 ID，则创建与文本长度相同的空 ID 列表
        if ids is None:
            ids = [None] * len(texts_or_text_and_labels)
        # 如果没有提供标签，则创建与文本长度相同的空标签列表
        if labels is None:
            labels = [None] * len(texts_or_text_and_labels)
        # 初始化示例列表和已添加标签集合
        examples = []
        added_labels = set()
        # 遍历文本、标签和 ID，创建 InputExample 对象并添加到示例列表中
        for text_or_text_and_label, label, guid in zip(texts_or_text_and_labels, labels, ids):
            if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
                text, label = text_or_text_and_label
            else:
                text = text_or_text_and_label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        # 更新示例
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # 更新标签
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            self.labels = list(set(self.labels).union(added_labels))

        return self.examples

    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
```