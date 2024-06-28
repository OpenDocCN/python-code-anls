# `.\data\processors\utils.py`

```py
# 导入所需的库和模块：csv、dataclasses、json，以及从typing中导入List、Optional和Union
import csv
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional, Union

# 导入日志记录工具，这里的logging来自于上层utils模块
from ...utils import is_tf_available, is_torch_available, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


@dataclass
class InputExample:
    """
    用于简单序列分类的单个训练/测试示例。

    Args:
        guid: 示例的唯一标识符。
        text_a: 字符串。第一个序列的未分词文本。对于单序列任务，只需指定此序列。
        text_b: (可选) 字符串。第二个序列的未分词文本。仅对序列对任务必须指定。
        label: (可选) 示例的标签。对于训练和开发示例应指定，但对于测试示例不应指定。
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """将该实例序列化为JSON字符串。"""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    单个数据特征集合。属性名称与模型输入的相应名称相同。

    Args:
        input_ids: 序列标记在词汇表中的索引。
        attention_mask: 避免对填充标记索引执行注意力的掩码。
            掩码值选择在 `[0, 1]` 范围内：通常为 `1` 表示非MASKED的标记， `0` 表示MASKED的标记（填充）。
        token_type_ids: (可选) 段标记索引，指示输入的第一和第二部分。只有某些模型会使用。
        label: (可选) 输入对应的标签。对于分类问题为整数，对于回归问题为浮点数。
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """将该实例序列化为JSON字符串。"""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DataProcessor:
    """用于序列分类数据集的数据转换器的基类。"""
    # 抽象方法，子类需要实现从一个包含 TensorFlow 张量的字典中获取一个示例
    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    # 抽象方法，子类需要实现从训练数据目录中获取一个 [`InputExample`] 集合
    def get_train_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    # 抽象方法，子类需要实现从开发数据目录中获取一个 [`InputExample`] 集合
    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    # 抽象方法，子类需要实现从测试数据目录中获取一个 [`InputExample`] 集合
    def get_test_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    # 抽象方法，子类需要实现获取数据集的标签列表
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # 方法用于将 TensorFlow 数据集的示例转换为 GLUE 数据集的正确格式
    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        # 如果数据集有多个标签，则将示例的标签转换为标签列表中对应的索引值
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    # 读取一个以制表符分隔的值文件（TSV 文件）
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    # 单句分类数据集的通用处理器类，继承自DataProcessor基类
    class SingleSentenceClassificationProcessor(DataProcessor):
        
        """Generic processor for a single sentence classification data set."""

        def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
            # 初始化处理器对象，设置标签、示例、处理模式和详细输出选项
            self.labels = [] if labels is None else labels
            self.examples = [] if examples is None else examples
            self.mode = mode
            self.verbose = verbose

        def __len__(self):
            # 返回处理器中示例的数量
            return len(self.examples)

        def __getitem__(self, idx):
            # 根据索引获取单个示例或切片，返回新的SingleSentenceClassificationProcessor对象
            if isinstance(idx, slice):
                return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
            return self.examples[idx]

        @classmethod
        def create_from_csv(
            cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
        ):
            # 从CSV文件创建处理器的类方法，返回新的处理器对象
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
            # 从示例文本或文本与标签列表创建处理器的类方法，返回新的处理器对象
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
            # 从CSV文件中读取行，并根据指定的列提取文本、标签和ID
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
        ):
            # 向处理器中添加示例文本、标签和可选ID，可以选择是否覆盖已有的标签和示例
    ):
        # 检查标签和文本或文本与标签列表的长度是否匹配，如果标签不为空且长度不一致则引发 ValueError 异常
        if labels is not None and len(texts_or_text_and_labels) != len(labels):
            raise ValueError(
                f"Text and labels have mismatched lengths {len(texts_or_text_and_labels)} and {len(labels)}"
            )
        # 检查 IDs 和文本或文本与标签列表的长度是否匹配，如果 IDs 不为空且长度不一致则引发 ValueError 异常
        if ids is not None and len(texts_or_text_and_labels) != len(ids):
            raise ValueError(f"Text and ids have mismatched lengths {len(texts_or_text_and_labels)} and {len(ids)}")
        # 如果 IDs 为空，则用 None 填充与文本或文本与标签列表长度相同的列表
        if ids is None:
            ids = [None] * len(texts_or_text_and_labels)
        # 如果标签为空，则用 None 填充与文本或文本与标签列表长度相同的列表
        if labels is None:
            labels = [None] * len(texts_or_text_and_labels)
        # 初始化空列表 examples，用于存储处理后的文本示例
        examples = []
        # 初始化集合 added_labels，用于存储添加的标签，确保每个标签只添加一次
        added_labels = set()
        # 遍历文本或文本与标签列表、标签列表和 IDs 列表的并行组合
        for text_or_text_and_label, label, guid in zip(texts_or_text_and_labels, labels, ids):
            # 如果文本或文本与标签是元组或列表且标签为空，则将其解包赋值给 text 和 label
            if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
                text, label = text_or_text_and_label
            else:
                text = text_or_text_and_label
            # 将当前标签添加到 added_labels 集合中
            added_labels.add(label)
            # 创建一个 InputExample 实例，并将其添加到 examples 列表中
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))

        # 更新 examples 列表
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # 更新 labels 列表
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            # 将当前 labels 列表与 added_labels 集合的并集转换为列表，更新 self.labels
            self.labels = list(set(self.labels).union(added_labels))

        # 返回更新后的 examples 列表
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