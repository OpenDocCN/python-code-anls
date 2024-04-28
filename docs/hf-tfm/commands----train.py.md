# `.\transformers\commands\train.py`

```
# 导入必要的模块
import os  # 导入操作系统模块
from argparse import ArgumentParser, Namespace  # 导入参数解析相关模块

# 从相应模块中导入类和函数
from ..data import SingleSentenceClassificationProcessor as Processor  # 导入数据处理器
from ..pipelines import TextClassificationPipeline  # 导入文本分类管道
from ..utils import is_tf_available, is_torch_available, logging  # 导入工具函数和判断模块是否可用的函数
from . import BaseTransformersCLICommand  # 从当前目录的子模块中导入基础命令类

# 如果既没有 TensorFlow 也没有 PyTorch，抛出运行时错误
if not is_tf_available() and not is_torch_available():
    raise RuntimeError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TensorFlow 训练参数
USE_XLA = False  # 是否启用 XLA 编译器
USE_AMP = False  # 是否启用自动混合精度训练


def train_command_factory(args: Namespace):
    """
    工厂函数，根据给定的命令行参数实例化训练命令。

    参数：
        args (Namespace)：命令行参数命名空间

    返回：
        TrainCommand：训练命令实例
    """
    return TrainCommand(args)


class TrainCommand(BaseTransformersCLICommand):
    @staticmethod
    # 将该命令注册到 argparse，以便在 transformer-cli 中可用

    # 创建一个子命令解析器，用于处理训练相关的命令
    train_parser = parser.add_parser("train", help="CLI tool to train a model on a task.")

    # 添加训练数据的参数
    train_parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="path to train (and optionally evaluation) dataset as a csv with tab separated labels and sentences.",
    )

    # 添加训练数据中标签所在列的参数
    train_parser.add_argument(
        "--column_label", type=int, default=0, help="Column of the dataset csv file with example labels."
    )

    # 添加训练数据中文本内容所在列的参数
    train_parser.add_argument(
        "--column_text", type=int, default=1, help="Column of the dataset csv file with example texts."
    )

    # 添加训练数据中样本 ID 所在列的参数
    train_parser.add_argument(
        "--column_id", type=int, default=2, help="Column of the dataset csv file with example ids."
    )

    # 添加是否跳过 CSV 文件的第一行（标题行）的参数
    train_parser.add_argument("--skip_first_row", action="store_true", help="Skip the first row of the csv file (headers).")

    # 添加验证数据的参数
    train_parser.add_argument("--validation_data", type=str, default="", help="path to validation dataset.")

    # 添加验证数据的划分比例的参数
    train_parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="if validation dataset is not provided, fraction of train dataset to use as validation dataset.",
    )

    # 添加模型输出路径的参数
    train_parser.add_argument("--output", type=str, default="./", help="path to saved the trained model.")

    # 添加训练任务类型的参数
    train_parser.add_argument(
        "--task", type=str, default="text_classification", help="Task to train the model on."
    )

    # 添加模型类型或路径的参数
    train_parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="Model's name or path to stored model."
    )

    # 添加训练时的批量大小参数
    train_parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")

    # 添加验证时的批量大小参数
    train_parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation.")

    # 添加学习率参数
    train_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")

    # 添加 Adam 优化器的 epsilon 参数
    train_parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam optimizer.")

    # 设置命令默认执行的函数为 train_command_factory
    train_parser.set_defaults(func=train_command_factory)
    # 初始化函数，接受参数 Namespace 对象
    def __init__(self, args: Namespace):
        # 获取日志记录器对象
        self.logger = logging.get_logger("transformers-cli/training")

        # 根据是否有 TensorFlow 可用确定框架类型
        self.framework = "tf" if is_tf_available() else "torch"

        # 创建输出目录（如果不存在）
        os.makedirs(args.output, exist_ok=True)
        self.output = args.output

        # 设置列名变量
        self.column_label = args.column_label
        self.column_text = args.column_text
        self.column_id = args.column_id

        # 打印加载任务和模型信息
        self.logger.info(f"Loading {args.task} pipeline for {args.model}")
        # 根据任务类型加载相应的 pipeline
        if args.task == "text_classification":
            self.pipeline = TextClassificationPipeline.from_pretrained(args.model)
        elif args.task == "token_classification":
            raise NotImplementedError
        elif args.task == "question_answering":
            raise NotImplementedError

        # 打印加载训练数据集信息
        self.logger.info(f"Loading dataset from {args.train_data}")
        # 从 CSV 文件创建训练数据集 Processor 对象
        self.train_dataset = Processor.create_from_csv(
            args.train_data,
            column_label=args.column_label,
            column_text=args.column_text,
            column_id=args.column_id,
            skip_first_row=args.skip_first_row,
        )
        self.valid_dataset = None
        # 如果有验证数据集，加载验证数据集信息
        if args.validation_data:
            self.logger.info(f"Loading validation dataset from {args.validation_data}")
            # 从 CSV 文件创建验证数据集 Processor 对象
            self.valid_dataset = Processor.create_from_csv(
                args.validation_data,
                column_label=args.column_label,
                column_text=args.column_text,
                column_id=args.column_id,
                skip_first_row=args.skip_first_row,
            )

        # 设置验证集拆分比例、训练批次大小、验证批次大小、学习率和 Adam 优化器 epsilon
        self.validation_split = args.validation_split
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon

    # 运行函数，根据框架类型调用不同的运行函数
    def run(self):
        if self.framework == "tf":
            return self.run_tf()
        return self.run_torch()

    # Torch 框架运行函数，抛出未实现异常
    def run_torch(self):
        raise NotImplementedError

    # TensorFlow 框架运行函数
    def run_tf(self):
        # 使用 pipeline 对象拟合训练数据集
        self.pipeline.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            validation_split=self.validation_split,
            learning_rate=self.learning_rate,
            adam_epsilon=self.adam_epsilon,
            train_batch_size=self.train_batch_size,
            valid_batch_size=self.valid_batch_size,
        )

        # 保存训练好的 pipeline
        self.pipeline.save_pretrained(self.output)
```