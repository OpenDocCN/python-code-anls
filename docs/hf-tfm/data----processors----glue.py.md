# `.\transformers\data\processors\glue.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归属于 Google AI Language Team 作者和 HuggingFace Inc. 团队以及 NVIDIA 公司
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发的软件，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" GLUE processors and helpers"""

# 导入所需的库
import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

# 导入所需的模块
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures

# 如果 TensorFlow 可用，则导入 TensorFlow 库
if is_tf_available():
    import tensorflow as tf

# 获取日志记录器
logger = logging.get_logger(__name__)

# 警告信息
DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
)

# 将输入的示例转换为特征
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
    # 发出警告
    warnings.warn(DEPRECATION_WARNING.format("function"), FutureWarning)
    # 如果 TensorFlow 可用且输入的示例是 tf.data.Dataset 类型，则调用 _tf_glue_convert_examples_to_features 函数
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    # 调用_glue_convert_examples_to_features函数，将示例转换为特征
    return _glue_convert_examples_to_features(
        # 传递给函数的示例列表
        examples,
        # 分词器，用于将示例转换为特征
        tokenizer,
        # 最大长度限制，用于截断或填充输入序列的长度
        max_length=max_length,
        # 任务类型，指定数据集的任务类型
        task=task,
        # 标签列表，包含数据集中所有可能的标签
        label_list=label_list,
        # 输出模式，指定模型的输出格式
        output_mode=output_mode
    )
if is_tf_available():
    # 如果 TensorFlow 可用，则定义一个函数用于将示例转换为特征
    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset,
        tokenizer: PreTrainedTokenizer,
        task=str,
        max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A `tf.data.Dataset` containing the task-specific features.
        """
        # 根据任务类型获取处理器
        processor = glue_processors[task]()
        # 将示例转换为处理器可处理的格式
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        # 将示例转换为特征
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
        # 根据任务类型确定标签类型
        label_type = tf.float32 if task == "sts-b" else tf.int64

        def gen():
            # 生成器函数，将特征转换为字典和标签
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = tokenizer.model_input_names

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
    # 如果未指定最大长度，则使用 tokenizer 的最大长度
    if max_length is None:
        max_length = tokenizer.model_max_length

    if task is not None:
        # 根据任务类型获取处理器
        processor = glue_processors[task]()
        if label_list is None:
            # 如果标签列表未指定，则获取处理器的标签列表
            label_list = processor.get_labels()
            logger.info(f"Using label list {label_list} for task {task}")
        if output_mode is None:
            # 如果输出模式未指定，则使用任务对应的输出模式
            output_mode = glue_output_modes[task]
            logger.info(f"Using output mode {output_mode} for task {task}")

    # 创建标签映射
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

    # 获取所有示例的标签
    labels = [label_from_example(example) for example in examples]

    # 使用 tokenizer 对示例进行编码
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # 打印前五个示例的信息
    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info(f"guid: {example.guid}")
        logger.info(f"features: {features[i]}")

    return features


class OutputMode(Enum):
    # 定义输出模式的枚举类
    classification = "classification"
    regression = "regression"
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 发出警告，提示该方法即将被弃用
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
        # 返回标签列表
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
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
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)
        # 发出警告，提示该方法即将被弃用
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
        # 获取开发集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 获取测试集示例
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")
    # 获取标签列表，包括"contradiction", "entailment", "neutral"
    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    # 为训练、开发和测试集创建示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        # 遍历每一行数据
        for i, line in enumerate(lines):
            # 跳过第一行
            if i == 0:
                continue
            # 生成唯一标识符
            guid = f"{set_type}-{line[0]}"
            # 获取文本 A
            text_a = line[8]
            # 获取文本 B
            text_b = line[9]
            # 如果是测试集，则标签为 None；否则为最后一列的标签
            label = None if set_type.startswith("test") else line[-1]
            # 将示例添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples
class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出关于处理器过时的警告
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 调用内部方法_create_examples，传入dev_mismatched.tsv文件的内容和数据集名称
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 调用内部方法_create_examples，传入test_mismatched.tsv文件的内容和数据集名称
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出关于处理器过时的警告
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取示例，包括索引、句子和标签信息
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        # 调用内部方法_create_examples，传入train.tsv文件的内容和数据集名称
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # 调用内部方法_create_examples，传入dev.tsv文件的内容和数据集名称
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # 调用内部方法_create_examples，传入test.tsv文件的内容和数据集名称
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # 返回标签列表
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 判断是否处于测试模式
        test_mode = set_type == "test"
        # 如果处于测试模式，从第二行开始处理
        if test_mode:
            lines = lines[1:]
        # 如果处于测试模式，文本索引为1，否则为3
        text_index = 1 if test_mode else 3
        examples = []
        # 遍历行数，创建InputExample实例并添加到examples列表中
        for i, line in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出关于处理器过时的警告
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取示例，包括索引、句子和标签信息
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
    # 获取训练集样本
    def get_train_examples(self, data_dir):
        """See base class."""
        # 调用_read_tsv方法读取train.tsv文件内容，然后调用_create_examples方法创建样本
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    # 获取开发集样本
    def get_dev_examples(self, data_dir):
        """See base class."""
        # 调用_read_tsv方法读取dev.tsv文件内容，然后调用_create_examples方法创建样本
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # 获取测试集样本
    def get_test_examples(self, data_dir):
        """See base class."""
        # 调用_read_tsv方法读取test.tsv文件内容，然后调用_create_examples方法创建样本
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    # 获取标签
    def get_labels(self):
        """See base class."""
        # 返回标签列表
        return ["0", "1"]

    # 创建样本
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 初始化样本列表
        examples = []
        # 如果是测试集，则text_index为1，否则为0
        text_index = 1 if set_type == "test" else 0
        # 遍历数据行
        for i, line in enumerate(lines):
            # 跳过表头行
            if i == 0:
                continue
            # 创建全局唯一标识符
            guid = f"{set_type}-{i}"
            # 获取文本内容
            text_a = line[text_index]
            # 如果是测试集，则标签为None，否则为行的第二个元素
            label = None if set_type == "test" else line[1]
            # 创建InputExample对象，并添加到样本列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        # 返回样本列表
        return examples
# STS-B 数据集处理器类，用于处理 GLUE 版本的数据集
class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    # 初始化方法，继承父类的初始化方法，并给出警告
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    # 从张量字典中获取示例，返回 InputExample 对象
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    # 获取训练集示例
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    # 获取验证集示例
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # 获取测试集示例
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    # 获取标签列表
    def get_labels(self):
        """See base class."""
        return [None]

    # 创建示例方法，用于训练、验证和测试集
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            # 使用示例的索引和集合类型生成 GUID
            guid = f"{set_type}-{line[0]}"
            # 获取文本 A 和文本 B
            text_a = line[7]
            text_b = line[8]
            # 如果是测试集，则标签为 None，否则为最后一列
            label = None if set_type == "test" else line[-1]
            # 添加 InputExample 对象到示例列表
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# QQP 数据集处理器类，用于处理 GLUE 版本的数据集
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    # 初始化方法，继承父类的初始化方法，并给出警告
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    # 从张量字典中获取示例，返回 InputExample 对象
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    # 获取训练集示例
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    # 获取验证集示例
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # 获取测试集示例
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    # 获取标签列表
    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    # 创建训练、开发和测试集的示例
    def _create_examples(self, lines, set_type):
        # 检查是否为测试集
        test_mode = set_type == "test"
        # 根据测试模式确定问题1和问题2的索引
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        # 遍历每一行数据
        for i, line in enumerate(lines):
            # 跳过第一行（标题行）
            if i == 0:
                continue
            # 生成示例的唯一标识符
            guid = f"{set_type}-{line[0]}"
            try:
                # 获取问题1和问题2的文本内容
                text_a = line[q1_index]
                text_b = line[q2_index]
                # 如果不是测试集，则获取标签
                label = None if test_mode else line[5]
            except IndexError:
                # 如果索引错误，则跳过该行
                continue
            # 将示例添加到列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples
class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出警告，提醒使用者该方法即将被弃用
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # 从张量字典中获取示例
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
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
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出警告，提醒使用者该方法即将被弃用
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
        return ["entailment", "not_entailment"]
    # 为给定的行列表创建训练、开发和测试集的示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # 初始化示例列表
        examples = []
        # 遍历行列表，创建示例
        for i, line in enumerate(lines):
            # 如果是第一行（索引为0），跳过该行，因为它是标题行
            if i == 0:
                continue
            # 根据行索引和集合类型生成示例的全局唯一标识符
            guid = f"{set_type}-{line[0]}"
            # 获取第一段文本
            text_a = line[1]
            # 获取第二段文本
            text_b = line[2]
            # 如果是测试集，标签为None；否则，从行列表中获取标签
            label = None if set_type == "test" else line[-1]
            # 创建一个输入示例并将其添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples
class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""
    
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 发出关于过时警告的警告
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
        # 创建训练、开发和测试集的示例
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
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
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}
```