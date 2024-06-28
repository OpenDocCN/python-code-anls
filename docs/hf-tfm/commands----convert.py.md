# `.\commands\convert.py`

```py
# 导入必要的模块和类
from argparse import ArgumentParser, Namespace  # 导入参数解析相关模块
from ..utils import logging  # 导入日志工具
from . import BaseTransformersCLICommand  # 导入基础转换器命令类

# 转换命令工厂函数，用于创建 ConvertCommand 实例
def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ServeCommand
    """
    return ConvertCommand(
        args.model_type, args.tf_checkpoint, args.pytorch_dump_output, args.config, args.finetuning_task_name
    )

# 当 transformers 无法导入时显示的错误消息
IMPORT_ERROR_MESSAGE = """
transformers can only be used from the commandline to convert TensorFlow models in PyTorch, In that case, it requires
TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.
"""

# ConvertCommand 类，继承自 BaseTransformersCLICommand
class ConvertCommand(BaseTransformersCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        # 添加转换命令到参数解析器
        train_parser = parser.add_parser(
            "convert",
            help="CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.",
        )
        # 添加转换命令的参数
        train_parser.add_argument("--model_type", type=str, required=True, help="Model's type.")
        train_parser.add_argument(
            "--tf_checkpoint", type=str, required=True, help="TensorFlow checkpoint path or folder."
        )
        train_parser.add_argument(
            "--pytorch_dump_output", type=str, required=True, help="Path to the PyTorch saved model output."
        )
        train_parser.add_argument("--config", type=str, default="", help="Configuration file path or folder.")
        train_parser.add_argument(
            "--finetuning_task_name",
            type=str,
            default=None,
            help="Optional fine-tuning task name if the TF model was a finetuned model.",
        )
        train_parser.set_defaults(func=convert_command_factory)

    def __init__(
        self,
        model_type: str,
        tf_checkpoint: str,
        pytorch_dump_output: str,
        config: str,
        finetuning_task_name: str,
        *args,
        ):
        # 获取名为 "transformers-cli/converting" 的日志记录器实例
        self._logger = logging.get_logger("transformers-cli/converting")

        # 记录信息日志，显示加载模型的信息
        self._logger.info(f"Loading model {model_type}")
        
        # 设置实例变量来存储模型类型
        self._model_type = model_type
        
        # 设置实例变量来存储 TensorFlow 的检查点路径
        self._tf_checkpoint = tf_checkpoint
        
        # 设置实例变量来存储 PyTorch 转换后的输出路径
        self._pytorch_dump_output = pytorch_dump_output
        
        # 设置实例变量来存储模型的配置信息
        self._config = config
        
        # 设置实例变量来存储微调任务的名称
        self._finetuning_task_name = finetuning_task_name
```