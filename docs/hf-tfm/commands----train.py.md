# `.\commands\train.py`

```py
# 导入标准库中的os模块，用于处理操作系统相关的功能
import os
# 从argparse模块中导入ArgumentParser类和Namespace类，用于处理命令行参数
from argparse import ArgumentParser, Namespace

# 从..data包中导入SingleSentenceClassificationProcessor作为Processor
from ..data import SingleSentenceClassificationProcessor as Processor
# 从..pipelines包中导入TextClassificationPipeline，用于文本分类任务的流水线处理
from ..pipelines import TextClassificationPipeline
# 从..utils包中导入is_tf_available、is_torch_available、logging等工具函数和类
from ..utils import is_tf_available, is_torch_available, logging
# 从当前包的__init__.py中导入BaseTransformersCLICommand类
from . import BaseTransformersCLICommand

# 如果既没有安装TensorFlow也没有安装PyTorch，则抛出运行时异常
if not is_tf_available() and not is_torch_available():
    raise RuntimeError("At least one of PyTorch or TensorFlow 2.0+ should be installed to use CLI training")

# TF训练参数设置
USE_XLA = False  # 是否使用XLA加速（TensorFlow专用）
USE_AMP = False  # 是否使用混合精度训练（TensorFlow专用）

def train_command_factory(args: Namespace):
    """
    工厂函数，根据给定的命令行参数实例化训练命令对象。

    Returns:
        TrainCommand: 实例化的训练命令对象
    """
    return TrainCommand(args)

class TrainCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        # 创建子命令 'train'，用于训练模型
        train_parser = parser.add_parser("train", help="CLI tool to train a model on a task.")

        # 添加训练数据路径参数
        train_parser.add_argument(
            "--train_data",
            type=str,
            required=True,
            help="path to train (and optionally evaluation) dataset as a csv with tab separated labels and sentences.",
        )

        # 添加数据集中标签所在列的参数
        train_parser.add_argument(
            "--column_label", type=int, default=0, help="Column of the dataset csv file with example labels."
        )

        # 添加数据集中文本所在列的参数
        train_parser.add_argument(
            "--column_text", type=int, default=1, help="Column of the dataset csv file with example texts."
        )

        # 添加数据集中ID所在列的参数
        train_parser.add_argument(
            "--column_id", type=int, default=2, help="Column of the dataset csv file with example ids."
        )

        # 添加是否跳过CSV文件第一行（标题行）的参数
        train_parser.add_argument(
            "--skip_first_row", action="store_true", help="Skip the first row of the csv file (headers)."
        )

        # 添加验证数据集路径参数
        train_parser.add_argument("--validation_data", type=str, default="", help="path to validation dataset.")

        # 添加验证数据集分割比例参数
        train_parser.add_argument(
            "--validation_split",
            type=float,
            default=0.1,
            help="if validation dataset is not provided, fraction of train dataset to use as validation dataset.",
        )

        # 添加保存训练模型的路径参数
        train_parser.add_argument("--output", type=str, default="./", help="path to saved the trained model.")

        # 添加训练任务类型参数
        train_parser.add_argument(
            "--task", type=str, default="text_classification", help="Task to train the model on."
        )

        # 添加模型名称或存储路径参数
        train_parser.add_argument(
            "--model", type=str, default="google-bert/bert-base-uncased", help="Model's name or path to stored model."
        )

        # 添加训练批次大小参数
        train_parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")

        # 添加验证批次大小参数
        train_parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation.")

        # 添加学习率参数
        train_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")

        # 添加Adam优化器的epsilon参数
        train_parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam optimizer.")

        # 设置训练命令的默认函数工厂
        train_parser.set_defaults(func=train_command_factory)
    # 初始化方法，接受一个参数 Namespace 类型的 args
    def __init__(self, args: Namespace):
        # 设置日志记录器，命名为 "transformers-cli/training"
        self.logger = logging.get_logger("transformers-cli/training")

        # 根据是否可用 TensorFlow 设置框架为 "tf" 或 "torch"
        self.framework = "tf" if is_tf_available() else "torch"

        # 创建输出目录，如果已存在则不创建
        os.makedirs(args.output, exist_ok=True)
        self.output = args.output  # 设置输出目录路径

        # 设置用于标签、文本和ID的列名
        self.column_label = args.column_label
        self.column_text = args.column_text
        self.column_id = args.column_id

        # 记录加载任务和模型信息到日志
        self.logger.info(f"Loading {args.task} pipeline for {args.model}")

        # 根据任务类型加载不同的pipeline
        if args.task == "text_classification":
            self.pipeline = TextClassificationPipeline.from_pretrained(args.model)
        elif args.task == "token_classification":
            raise NotImplementedError  # 抛出未实现错误
        elif args.task == "question_answering":
            raise NotImplementedError  # 抛出未实现错误

        # 记录加载训练数据集信息到日志
        self.logger.info(f"Loading dataset from {args.train_data}")

        # 从CSV文件创建训练数据集对象，使用指定的列名和参数
        self.train_dataset = Processor.create_from_csv(
            args.train_data,
            column_label=args.column_label,
            column_text=args.column_text,
            column_id=args.column_id,
            skip_first_row=args.skip_first_row,
        )

        # 初始化验证数据集为 None
        self.valid_dataset = None

        # 如果指定了验证数据集路径，则加载验证数据集信息到日志
        if args.validation_data:
            self.logger.info(f"Loading validation dataset from {args.validation_data}")

            # 从CSV文件创建验证数据集对象，使用指定的列名和参数
            self.valid_dataset = Processor.create_from_csv(
                args.validation_data,
                column_label=args.column_label,
                column_text=args.column_text,
                column_id=args.column_id,
                skip_first_row=args.skip_first_row,
            )

        # 设置验证集分割比例、训练批次大小、验证批次大小、学习率和Adam优化器的epsilon值
        self.validation_split = args.validation_split
        self.train_batch_size = args.train_batch_size
        self.valid_batch_size = args.valid_batch_size
        self.learning_rate = args.learning_rate
        self.adam_epsilon = args.adam_epsilon

    # 运行方法，根据框架类型调用相应的运行方法
    def run(self):
        if self.framework == "tf":
            return self.run_tf()  # 调用 TensorFlow 版本的运行方法
        return self.run_torch()  # 调用 PyTorch 版本的运行方法

    # 用于在 PyTorch 框架下运行的方法，抛出未实现错误
    def run_torch(self):
        raise NotImplementedError

    # 用于在 TensorFlow 框架下运行的方法，训练 pipeline 模型并保存
    def run_tf(self):
        # 使用训练数据集训练 pipeline 模型，同时指定验证数据集和训练参数
        self.pipeline.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            validation_split=self.validation_split,
            learning_rate=self.learning_rate,
            adam_epsilon=self.adam_epsilon,
            train_batch_size=self.train_batch_size,
            valid_batch_size=self.valid_batch_size,
        )

        # 将训练好的 pipeline 模型保存到指定的输出目录
        self.pipeline.save_pretrained(self.output)
```