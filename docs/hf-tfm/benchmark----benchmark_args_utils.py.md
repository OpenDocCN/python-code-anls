# `.\benchmark\benchmark_args_utils.py`

```py
# 导入必要的模块和库
import dataclasses  # 导入用于定义数据类的模块
import json  # 导入处理 JSON 数据的模块
import warnings  # 导入警告处理模块
from dataclasses import dataclass, field  # 从 dataclasses 模块导入 dataclass 装饰器和 field 函数
from time import time  # 从 time 模块导入 time 函数
from typing import List  # 导入 List 类型提示

from ..utils import logging  # 导入相对路径的 logging 模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def list_field(default=None, metadata=None):
    # 返回一个数据类 field，用于处理列表类型的字段
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class BenchmarkArguments:
    """
    BenchMarkArguments are arguments we use in our benchmark scripts **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    models: List[str] = list_field(
        default=[],
        metadata={
            "help": (
                "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version"
                " of all available models"
            )
        },
    )

    batch_sizes: List[int] = list_field(
        default=[8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"}
    )

    sequence_lengths: List[int] = list_field(
        default=[8, 32, 128, 512],
        metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"},
    )

    inference: bool = field(
        default=True,
        metadata={"help": "Whether to benchmark inference of model. Inference can be disabled via --no-inference."},
    )
    cuda: bool = field(
        default=True,
        metadata={"help": "Whether to run on available cuda devices. Cuda can be disabled via --no-cuda."},
    )
    tpu: bool = field(
        default=True, metadata={"help": "Whether to run on available tpu devices. TPU can be disabled via --no-tpu."}
    )
    fp16: bool = field(default=False, metadata={"help": "Use FP16 to accelerate inference."})
    training: bool = field(default=False, metadata={"help": "Benchmark training of model"})
    verbose: bool = field(default=False, metadata={"help": "Verbose memory tracing"})
    speed: bool = field(
        default=True,
        metadata={"help": "Whether to perform speed measurements. Speed measurements can be disabled via --no-speed."},
    )
    memory: bool = field(
        default=True,
        metadata={
            "help": "Whether to perform memory measurements. Memory measurements can be disabled via --no-memory"
        },
    )
    # 设置一个布尔类型的字段，用于指示是否进行内存测量，可以通过 --no-memory 参数禁用内存测量

    trace_memory_line_by_line: bool = field(default=False, metadata={"help": "Trace memory line by line"})
    # 设置一个布尔类型的字段，用于指示是否逐行跟踪内存使用情况

    save_to_csv: bool = field(default=False, metadata={"help": "Save result to a CSV file"})
    # 设置一个布尔类型的字段，用于指示是否将结果保存到 CSV 文件中

    log_print: bool = field(default=False, metadata={"help": "Save all print statements in a log file"})
    # 设置一个布尔类型的字段，用于指示是否将所有的打印语句保存到日志文件中

    env_print: bool = field(default=False, metadata={"help": "Whether to print environment information"})
    # 设置一个布尔类型的字段，用于指示是否打印环境信息

    multi_process: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use multiprocessing for memory and speed measurement. It is highly recommended to use"
                " multiprocessing for accurate CPU and GPU memory measurements. This option should only be disabled"
                " for debugging / testing and on TPU."
            )
        },
    )
    # 设置一个布尔类型的字段，用于指示是否使用多进程进行内存和速度测量，建议用于准确的 CPU 和 GPU 内存测量，仅在调试/测试和使用 TPU 时禁用此选项

    inference_time_csv_file: str = field(
        default=f"inference_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv."},
    )
    # 设置一个字符串类型的字段，指定保存推理时间结果的 CSV 文件名

    inference_memory_csv_file: str = field(
        default=f"inference_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv."},
    )
    # 设置一个字符串类型的字段，指定保存推理内存结果的 CSV 文件名

    train_time_csv_file: str = field(
        default=f"train_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv for training."},
    )
    # 设置一个字符串类型的字段，指定保存训练时间结果的 CSV 文件名

    train_memory_csv_file: str = field(
        default=f"train_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv for training."},
    )
    # 设置一个字符串类型的字段，指定保存训练内存结果的 CSV 文件名

    env_info_csv_file: str = field(
        default=f"env_info_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving environment information."},
    )
    # 设置一个字符串类型的字段，指定保存环境信息的 CSV 文件名

    log_filename: str = field(
        default=f"log_{round(time())}.csv",
        metadata={"help": "Log filename used if print statements are saved in log."},
    )
    # 设置一个字符串类型的字段，指定保存打印语句的日志文件名

    repeat: int = field(default=3, metadata={"help": "Times an experiment will be run."})
    # 设置一个整数类型的字段，指定实验运行的次数

    only_pretrain_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Instead of loading the model as defined in `config.architectures` if exists, just load the pretrain"
                " model weights."
            )
        },
    )
    # 设置一个布尔类型的字段，用于指示是否仅加载预训练模型权重，而不加载 config.architectures 中定义的模型结构

    def __post_init__(self):
        warnings.warn(
            f"The class {self.__class__} is deprecated. Hugging Face Benchmarking utils"
            " are deprecated in general and it is advised to use external Benchmarking libraries "
            " to benchmark Transformer models.",
            FutureWarning,
        )
        # 初始化方法，发出警告提示类已过时，建议使用外部基准库对 Transformer 模型进行基准测试

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)
        # 将当前实例序列化为 JSON 字符串的方法
    # 返回模型名称列表，如果模型列表为空，则引发值错误异常
    def model_names(self) -> List[str]:
        # 检查模型列表是否为空
        if len(self.models) <= 0:
            # 如果为空，抛出值错误异常，提醒用户至少提供一个模型名称或模型标识符
            raise ValueError(
                "Please make sure you provide at least one model name / model identifier, *e.g.* `--models"
                " google-bert/bert-base-cased` or `args.models = ['google-bert/bert-base-cased']."
            )
        # 返回模型名称列表
        return self.models

    @property
    # 返回布尔值，指示是否进行多进程处理
    def do_multi_processing(self):
        # 如果不使用多进程，则返回 False
        if not self.multi_process:
            return False
        # 如果使用 TPU，则记录信息并返回 False，因为目前不支持在 TPU 上进行多进程处理
        elif self.is_tpu:
            logger.info("Multiprocessing is currently not possible on TPU.")
            return False
        else:
            # 否则返回 True，表示可以进行多进程处理
            return True
```