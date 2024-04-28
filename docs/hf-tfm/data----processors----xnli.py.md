# `.\transformers\data\processors\xnli.py`

```py
# 导入所需模块
import os

# 导入 logging 模块
from ...utils import logging
# 从当前目录下的 utils 模块导入 DataProcessor 和 InputExample 类
from .utils import DataProcessor, InputExample

# 获取 logger 对象
logger = logging.get_logger(__name__)


# 定义 XnliProcessor 类，继承自 DataProcessor 类
class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """

    # 初始化方法，接收 language 和 train_language 两个参数
    def __init__(self, language, train_language=None):
        # 设置语言属性
        self.language = language
        # 设置训练语言属性，如果未指定，则与 language 属性相同
        self.train_language = train_language

    # 获取训练样本的方法
    def get_train_examples(self, data_dir):
        """See base class."""
        # 如果 train_language 为 None，则 lg 取 language 的值，否则取 train_language 的值
        lg = self.language if self.train_language is None else self.train_language
        # 读取训练数据的 TSV 文件，并将内容存储在 lines 变量中
        lines = self._read_tsv(os.path.join(data_dir, f"XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv"))
        # 初始化 examples 列表，用于存储训练样本
        examples = []
        # 遍历 lines 列表中的每一行
        for i, line in enumerate(lines):
            # 如果是第一行则跳过
            if i == 0:
                continue
            # 生成唯一的样本标识符 guid
            guid = f"train-{i}"
            # 获取文本 A
            text_a = line[0]
            # 获取文本 B
            text_b = line[1]
            # 获取标签，并将 "contradictory" 转换为 "contradiction"
            label = "contradiction" if line[2] == "contradictory" else line[2]
            # 检查 text_a 是否为字符串，若不是则抛出 ValueError 异常
            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            # 检查 text_b 是否为字符串，若不是则抛出 ValueError 异常
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            # 检查 label 是否为字符串，若不是则抛出 ValueError 异常
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            # 将生成的 InputExample 对象添加到 examples 列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回 examples 列表
        return examples
    # 获取测试集示例的方法，从指定的数据目录读取数据
    def get_test_examples(self, data_dir):
        """See base class."""  # 查看基类方法的文档字符串
        # 读取指定路径下的 TSV 文件，返回行列表
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        # 初始化示例列表
        examples = []
        # 遍历行索引和行内容
        for i, line in enumerate(lines):
            # 如果是第一行，跳过
            if i == 0:
                continue
            # 获取语言标签
            language = line[0]
            # 如果语言不是指定的语言，跳过
            if language != self.language:
                continue
            # 生成示例的唯一标识符
            guid = f"test-{i}"
            # 获取文本 A
            text_a = line[6]
            # 获取文本 B
            text_b = line[7]
            # 获取标签
            label = line[1]
            # 如果文本 A 不是字符串，引发 ValueError 异常
            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            # 如果文本 B 不是字符串，引发 ValueError 异常
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            # 如果标签不是字符串，引发 ValueError 异常
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            # 将示例添加到示例列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回示例列表
        return examples

    # 获取标签列表的方法
    def get_labels(self):
        """See base class."""  # 查看基类方法的文档字符串
        # 返回标签列表
        return ["contradiction", "entailment", "neutral"]
# XNLI 数据集的处理器字典，将任务名映射到对应的处理器类
xnli_processors = {
    "xnli": XnliProcessor,
}

# XNLI 数据集的输出模式字典，将任务名映射到对应的输出模式（这里为分类）
xnli_output_modes = {
    "xnli": "classification",
}

# XNLI 数据集的任务及其对应的类别数字典，将任务名映射到对应的类别数（这里为3分类）
xnli_tasks_num_labels = {
    "xnli": 3,
}
```