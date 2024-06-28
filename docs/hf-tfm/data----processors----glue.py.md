# `.\data\processors\glue.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包括 Google AI Language Team 和 HuggingFace Inc. 团队的版权声明
# 版权声明，包括 NVIDIA CORPORATION 的版权声明
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证的要求，否则不得使用本文件
# 可以从以下链接获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 如果不符合适用法律或书面同意，则不得分发本软件
# 本软件基于"原样"提供，无任何明示或暗示的保证或条件
# 更多详细信息，请参阅许可证
""" GLUE processors and helpers"""

# 导入操作系统相关模块
import os
# 导入警告相关模块
import warnings
# 导入数据类相关模块
from dataclasses import asdict
# 导入枚举类型相关模块
from enum import Enum
# 导入列表、可选值和联合类型相关模块
from typing import List, Optional, Union

# 导入令牌化工具相关模块
from ...tokenization_utils import PreTrainedTokenizer
# 导入 TensorFlow 是否可用相关模块
from ...utils import is_tf_available, logging
# 导入数据处理器、输入示例、输入特征相关模块
from .utils import DataProcessor, InputExample, InputFeatures

# 如果 TensorFlow 可用，则导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf

# 获取日志记录器
logger = logging.get_logger(__name__)

# 警告信息：此函数将很快从库中移除，预处理应使用 🤗 Datasets 库处理
DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
)

# 函数：将输入示例转换为特征列表
def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of `InputFeatures`

    Args:
        examples: List of `InputExamples` or `tf.data.Dataset` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the `processor.get_labels()` method
        output_mode: String indicating the output mode. Either `regression` or `classification`

    Returns:
        If the `examples` input is a `tf.data.Dataset`, will return a `tf.data.Dataset` containing the task-specific
        features. If the input is a list of `InputExamples`, will return a list of task-specific `InputFeatures` which
        can be fed to the model.

    """
    # 发出警告：此函数将很快从库中移除
    warnings.warn(DEPRECATION_WARNING.format("function"), FutureWarning)
    # 如果 TensorFlow 可用且输入示例为 tf.data.Dataset 类型，则调用对应的 TensorFlow 版本的转换函数
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    # 调用一个函数来将示例转换为特征并返回结果
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )
if is_tf_available():
    # 如果 TensorFlow 可用，则定义一个私有函数 _tf_glue_convert_examples_to_features
    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        task=str,
        max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        将示例转换为特征集合的 TensorFlow 数据集。

        Returns:
            包含特定任务特征的 `tf.data.Dataset` 对象。
        """
        # 根据任务选择对应的处理器
        processor = glue_processors[task]()
        # 转换示例为 TensorFlow 数据集格式，并使用处理器处理每个示例
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        # 将处理后的示例转换为特征集合
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
        # 根据任务类型确定标签类型
        label_type = tf.float32 if task == "sts-b" else tf.int64

        def gen():
            # 生成器函数，为 TensorFlow 数据集生成特征和标签对
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = tokenizer.model_input_names

        # 返回基于生成器的 TensorFlow 数据集对象
        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, label_type),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        # 如果指定了任务，选择对应的处理器和标签列表
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    # 构建标签映射字典
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        # 根据示例获取标签
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    # 获取所有示例的标签列表
    labels = [label_from_example(example) for example in examples]

    # 使用 tokenizer 批量编码文本对
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    # 构建输入特征对象列表
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # 输出前五个示例的日志信息
    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example.guid}")
        logger.info(f"features: {features[i]}")

    return features


class OutputMode(Enum):
    # 定义输出模式枚举类
    classification = "classification"
    regression = "regression"
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 发出关于过时警告的警告消息
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取示例
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 获取训练集示例
        logger.info(f"LOOKING AT {os.path.join(data_dir, 'train.tsv')}")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 获取开发集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回数据集的标签列表
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 为训练、开发和测试集创建示例
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 发出关于过时警告的警告消息
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取示例
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 获取训练集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 获取匹配开发集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取匹配测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")
    # 返回一个包含标签列表的字符串数组，作为基类方法的实现
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    # 根据给定的行列表和集合类型创建示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            # 跳过第一行，因为它通常包含标题而非数据行
            if i == 0:
                continue
            # 使用行的第一个字段与集合类型结合创建唯一标识符
            guid = f"{set_type}-{line[0]}"
            # 获取文本 A，通常在行的第 8 列
            text_a = line[8]
            # 获取文本 B，通常在行的第 9 列
            text_b = line[9]
            # 如果是测试集合的一部分，标签置为 None；否则使用行的最后一列作为标签
            label = None if set_type.startswith("test") else line[-1]
            # 创建一个 InputExample 实例并将其添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回创建的示例列表
        return examples
class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类构造函数，并发出未来警告
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 读取 dev_mismatched.tsv 文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 读取 test_mismatched.tsv 文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类构造函数，并发出未来警告
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从 tensor 字典中提取数据并创建 InputExample 对象
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 读取 train.tsv 文件并创建训练示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 读取 dev.tsv 文件并创建 dev 示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 读取 test.tsv 文件并创建 test 示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回标签列表 ["0", "1"]
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 根据 set_type 创建相应数据集的示例
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]  # 如果是测试模式，跳过表头
        text_index = 1 if test_mode else 3  # 确定文本在行中的索引
        examples = []
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"  # 构建全局唯一标识符
            text_a = line[text_index]  # 获取文本 A
            label = None if test_mode else line[1]  # 获取标签（训练和验证集有标签，测试集没有）
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类构造函数，并发出未来警告
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从 tensor 字典中提取数据并创建 InputExample 对象
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
    # 从数据目录中读取 train.tsv 文件并创建训练集的示例
    def get_train_examples(self, data_dir):
        """See base class."""
        # 调用内部方法 _read_tsv 读取 train.tsv 文件内容，并调用 _create_examples 创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    # 从数据目录中读取 dev.tsv 文件并创建开发集的示例
    def get_dev_examples(self, data_dir):
        """See base class."""
        # 调用内部方法 _read_tsv 读取 dev.tsv 文件内容，并调用 _create_examples 创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # 从数据目录中读取 test.tsv 文件并创建测试集的示例
    def get_test_examples(self, data_dir):
        """See base class."""
        # 调用内部方法 _read_tsv 读取 test.tsv 文件内容，并调用 _create_examples 创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    # 返回数据集的标签，这里是二分类任务，标签为 ["0", "1"]
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # 根据给定的 lines 和数据集类型 set_type 创建示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        # 确定文本在行数据中的索引，对于测试集来说是第二列（index 1），对于其他是第一列（index 0）
        text_index = 1 if set_type == "test" else 0
        # 遍历每一行数据
        for i, line in enumerate(lines):
            # 跳过表头行（第一行）
            if i == 0:
                continue
            # 每个示例有一个全局唯一的 ID
            guid = f"{set_type}-{i}"
            # 获取文本内容，如果是测试集则直接取第一列文本，否则取第一列文本和第二列标签
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            # 创建 InputExample 对象并添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 发出关于过时警告的警告
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中创建输入示例对象
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取训练集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取开发集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取测试集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回数据集的标签列表
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            # 构建示例的唯一标识符
            guid = f"{set_type}-{line[0]}"
            text_a = line[7]  # 第一个文本字段
            text_b = line[8]  # 第二个文本字段
            label = None if set_type == "test" else line[-1]  # 标签，测试集时为None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 发出关于过时警告的警告
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中创建输入示例对象
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取训练集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取开发集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 从指定目录中读取测试集文件并创建示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回数据集的标签列表
        return ["0", "1"]
    # 定义一个方法用于创建训练、开发和测试集的样本示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 根据传入的set_type确定是否为测试模式
        test_mode = set_type == "test"
        # 根据测试模式确定问题1和问题2在每行数据中的索引位置
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        # 遍历所有行数据
        for i, line in enumerate(lines):
            # 跳过第一行（标题行）
            if i == 0:
                continue
            # 每个样本的全局唯一标识(guid)格式为"{set_type}-{line[0]}"
            guid = f"{set_type}-{line[0]}"
            try:
                # 获取问题1和问题2的文本内容
                text_a = line[q1_index]
                text_b = line[q2_index]
                # 如果是测试模式，标签为None；否则取出第5列作为标签
                label = None if test_mode else line[5]
            except IndexError:
                # 如果索引错误（行数据不足），跳过该行
                continue
            # 创建一个InputExample对象，并加入到examples列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回创建的样本示例列表
        return examples
class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，并发出未来警告
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取数据并创建输入示例
        return InputExample(
            tensor_dict["idx"].numpy(),  # 获取索引并转换为NumPy数组
            tensor_dict["question"].numpy().decode("utf-8"),  # 获取问题字符串并解码为UTF-8格式
            tensor_dict["sentence"].numpy().decode("utf-8"),  # 获取句子字符串并解码为UTF-8格式
            str(tensor_dict["label"].numpy()),  # 获取标签并转换为字符串
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 获取训练集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 获取验证集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回标签列表
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue  # 跳过标题行
            guid = f"{set_type}-{line[0]}"  # 创建全局唯一ID
            text_a = line[1]  # 获取第一个文本
            text_b = line[2]  # 获取第二个文本
            label = None if set_type == "test" else line[-1]  # 如果是测试集则标签为空，否则为最后一列
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))  # 添加输入示例
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，并发出未来警告
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取数据并创建输入示例
        return InputExample(
            tensor_dict["idx"].numpy(),  # 获取索引并转换为NumPy数组
            tensor_dict["sentence1"].numpy().decode("utf-8"),  # 获取第一个句子并解码为UTF-8格式
            tensor_dict["sentence2"].numpy().decode("utf-8"),  # 获取第二个句子并解码为UTF-8格式
            str(tensor_dict["label"].numpy()),  # 获取标签并转换为字符串
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 获取训练集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 获取验证集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回标签列表
        return ["entailment", "not_entailment"]
    # 创建用于训练、开发和测试集的示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 初始化空列表来存储示例
        examples = []
        # 遍历输入的每一行数据
        for i, line in enumerate(lines):
            # 跳过第一行（通常是标题行）
            if i == 0:
                continue
            # 根据数据集类型和行索引创建全局唯一标识符
            guid = f"{set_type}-{line[0]}"
            # 获取第一列文本作为 text_a
            text_a = line[1]
            # 获取第二列文本作为 text_b
            text_b = line[2]
            # 如果是测试集，标签设为 None；否则使用行数据的最后一列作为标签
            label = None if set_type == "test" else line[-1]
            # 创建一个输入示例并添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回创建的示例列表
        return examples
class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 根据张量字典创建输入示例对象
        return InputExample(
            tensor_dict["idx"].numpy(),  # 使用张量中的索引值并转换为 numpy 数组
            tensor_dict["sentence1"].numpy().decode("utf-8"),  # 将张量中的句子1数据转换为 UTF-8 编码字符串
            tensor_dict["sentence2"].numpy().decode("utf-8"),  # 将张量中的句子2数据转换为 UTF-8 编码字符串
            str(tensor_dict["label"].numpy()),  # 使用张量中的标签值并转换为字符串
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 获取训练集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 获取开发集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回标签列表
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"  # 使用数据集类型和行索引创建全局唯一标识符
            text_a = line[1]  # 获取第一个文本
            text_b = line[2]  # 获取第二个文本
            label = None if set_type == "test" else line[-1]  # 如果是测试集，标签设为 None；否则使用数据中的标签值
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

glue_processors = {
    "cola": ColaProcessor,  # 对应的处理器类
    "mnli": MnliProcessor,  # 对应的处理器类
    "mnli-mm": MnliMismatchedProcessor,  # 对应的处理器类
    "mrpc": MrpcProcessor,  # 对应的处理器类
    "sst-2": Sst2Processor,  # 对应的处理器类
    "sts-b": StsbProcessor,  # 对应的处理器类
    "qqp": QqpProcessor,  # 对应的处理器类
    "qnli": QnliProcessor,  # 对应的处理器类
    "rte": RteProcessor,  # 对应的处理器类
    "wnli": WnliProcessor,  # 对应的处理器类，本身就是 WnliProcessor 类
}

glue_output_modes = {
    "cola": "classification",  # 输出模式为分类
    "mnli": "classification",  # 输出模式为分类
    "mnli-mm": "classification",  # 输出模式为分类
    "mrpc": "classification",  # 输出模式为分类
    "sst-2": "classification",  # 输出模式为分类
    "sts-b": "regression",  # 输出模式为回归
    "qqp": "classification",  # 输出模式为分类
    "qnli": "classification",  # 输出模式为分类
    "rte": "classification",  # 输出模式为分类
    "wnli": "classification",  # 输出模式为分类
}
```