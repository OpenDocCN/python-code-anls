# `.\lucidrains\electra-pytorch\examples\glue\processors.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者和团队信息
# 版权声明，版权所有，保留所有权利
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
""" GLUE processors and helpers """

# 导入日志记录模块
import logging
# 导入操作系统模块
import os

# 导入自定义模块
# from ...file_utils import is_tf_available
from utils import DataProcessor, InputExample, InputFeatures

# 定义一个 lambda 函数，用于检查 TensorFlow 是否可用
is_tf_available = lambda: False

# 如果 TensorFlow 可用，则导入 TensorFlow 模块
if is_tf_available():
    import tensorflow as tf

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义函数，将示例转换为特征
def glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    # ���始化变量，用于检查是否为 TensorFlow 数据集
    is_tf_dataset = False
    # 如果 TensorFlow 可用且 examples 是 tf.data.Dataset 类型，则设置为 True
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    # 如果指定了任务，则创建对应的处理器
    if task is not None:
        processor = glue_processors[task]()
        # 如果标签列表为空，则从处理器中获取标签列表
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        # 如果输出模式为空，则从 GLUE 输出模式中获取
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    # 创建标签映射字典
    label_map = {label: i for i, label in enumerate(label_list)}

    # 初始化特征列表
    features = []
    # 遍历所有的例子，并获取索引和例子内容
    for (ex_index, example) in enumerate(examples):
        # 初始化例子的数量
        len_examples = 0
        # 如果是 TensorFlow 数据集
        if is_tf_dataset:
            # 从张量字典中获取例子
            example = processor.get_example_from_tensor_dict(example)
            # 对例子进行 TFDS 映射
            example = processor.tfds_map(example)
            # 获取例子的数量
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            # 获取例子的数量
            len_examples = len(examples)
        # 每处理 10000 个例子输出日志信息
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        # 使用分词器对文本进行编码
        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # 生成注意力掩码，用于指示哪些是真实标记，哪些是填充标记
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # 对序列进行零填充
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # 断言输入长度与最大长度相等
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        # 根据输出模式处理标签
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # 输出前5个例子的信息
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        # 将特征添加到列表中
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    # 如果 TensorFlow 可用且是 TensorFlow 数据集
    if is_tf_available() and is_tf_dataset:

        # 生成器函数，用于生成数据集
        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        # 从生成器创建 TensorFlow 数据集
        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    # 返回特征列表
    return features
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """获取标签列表。"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """为训练集和开发集创建示例。"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """获取标签列表。"""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """为训练集和开发集创建示例。"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """获取标签列表。"""
        return ["0", "1"]
    # 创建训练集和开发集的示例
    def _create_examples(self, lines, set_type):
        # 初始化示例列表
        examples = []
        # 遍历每一行数据
        for (i, line) in enumerate(lines):
            # 生成示例的唯一标识符
            guid = "%s-%s" % (set_type, i)
            # 获取文本 A 的内容
            text_a = line[3]
            # 获取标签
            label = line[1]
            # 将示例添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        # 返回示例列表
        return examples
# 定义处理 SST-2 数据集的 Processor 类
class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    # 从张量字典中获取示例
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
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

    # 获取标签
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # 创建训练集和验证集示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


# 定义处理 STS-B 数据集的 Processor 类
class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    # 从张量字典中获取示例
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

    # 获取标签
    def get_labels(self):
        """See base class."""
        return [None]

    # 创建训练集和验证集示例
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# 定义处理 QQP 数据集的 Processor 类
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    # 从张量字典中获取示例
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

    # 获取标签
    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    # 创建训练集和开发集的示例
    def _create_examples(self, lines, set_type):
        # 初始化示例列表
        examples = []
        # 遍历每一行数据
        for (i, line) in enumerate(lines):
            # 跳过第一行数据
            if i == 0:
                continue
            # 生成示例的唯一标识符
            guid = "%s-%s" % (set_type, line[0])
            # 尝试获取文本A、文本B和标签信息
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            # 如果索引超出范围，则跳过该行数据
            except IndexError:
                continue
            # 将示例添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples
class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched")

    def get_labels(self):
        """获取标签列表。"""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """为训练集和开发集创建示例。"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """获取标签列表。"""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """为训练集和开发集创建示例。"""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """从张量字典中获取示例。"""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """获取训练集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """获取开发集示例。"""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """获取标签列表。"""
        return ["0", "1"]
    # 创建训练集和验证集的示例
    def _create_examples(self, lines, set_type):
        # 初始化示例列表
        examples = []
        # 遍历每一行数据
        for (i, line) in enumerate(lines):
            # 跳过第一行数据
            if i == 0:
                continue
            # 生成示例的唯一标识符
            guid = "%s-%s" % (set_type, line[0])
            # 获取文本 A
            text_a = line[1]
            # 获取文本 B
            text_b = line[2]
            # 获取标签
            label = line[-1]
            # 创建输入示例对象并添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples
# 定义每个 GLUE 任务对应的标签数量
glue_tasks_num_labels = {
    "cola": 2,  # CoLA 任务有2个标签
    "mnli": 3,  # MNLI 任务有3个标签
    "mrpc": 2,  # MRPC 任务有2个标签
    "sst-2": 2,  # SST-2 任务有2个标签
    "sts-b": 1,  # STS-B 任务有1个标签
    "qqp": 2,  # QQP 任务有2个标签
    "qnli": 2,  # QNLI 任务有2个标签
    "rte": 2,  # RTE 任务有2个标签
    "wnli": 2,  # WNLI 任务有2个标签
}

# 定义每个 GLUE 任务对应的处理器类
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

# 定义每个 GLUE 任务对应的输出模式
glue_output_modes = {
    "cola": "classification",  # CoLA 任务的输出模式为分类
    "mnli": "classification",  # MNLI 任务的输出模式为分类
    "mnli-mm": "classification",  # MNLI-MM 任务的输出模式为分类
    "mrpc": "classification",  # MRPC 任务的输出模式为分类
    "sst-2": "classification",  # SST-2 任务的输出模式为分类
    "sts-b": "regression",  # STS-B 任务的输出模式为回归
    "qqp": "classification",  # QQP 任务的输出模式为分类
    "qnli": "classification",  # QNLI 任务的输出模式为分类
    "rte": "classification",  # RTE 任务的输出模式为分类
    "wnli": "classification",  # WNLI 任务的输出模式为分类
}
```