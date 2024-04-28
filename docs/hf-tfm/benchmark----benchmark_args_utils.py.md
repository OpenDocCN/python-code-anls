# `.\transformers\benchmark\benchmark_args_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 版权声明，版权归 2018 年 NVIDIA 公司所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块
import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from time import time
from typing import List

# 导入自定义的 logging 模块
from ..utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一个函数，用于创建列表字段
def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

# 定义一个数据类 BenchmarkArguments，用于存储与训练循环相关的基准测试参数
@dataclass
class BenchmarkArguments:
    """
    BenchMarkArguments are arguments we use in our benchmark scripts **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    # 模型名称列表，默认为空列表
    models: List[str] = list_field(
        default=[],
        metadata={
            "help": (
                "Model checkpoints to be provided to the AutoModel classes. Leave blank to benchmark the base version"
                " of all available models"
            )
        },
    )

    # 批量大小列表，默认为 [8]
    batch_sizes: List[int] = list_field(
        default=[8], metadata={"help": "List of batch sizes for which memory and time performance will be evaluated"}
    )

    # 序列长度列表，默认为 [8, 32, 128, 512]
    sequence_lengths: List[int] = list_field(
        default=[8, 32, 128, 512],
        metadata={"help": "List of sequence lengths for which memory and time performance will be evaluated"},
    )

    # 推理标志，默认为 True
    inference: bool = field(
        default=True,
        metadata={"help": "Whether to benchmark inference of model. Inference can be disabled via --no-inference."},
    )
    
    # CUDA 标志，默认为 True
    cuda: bool = field(
        default=True,
        metadata={"help": "Whether to run on available cuda devices. Cuda can be disabled via --no-cuda."},
    )
    
    # TPU 标志，默认为 True
    tpu: bool = field(
        default=True, metadata={"help": "Whether to run on available tpu devices. TPU can be disabled via --no-tpu."}
    )
    
    # FP16 标志，默认为 False
    fp16: bool = field(default=False, metadata={"help": "Use FP16 to accelerate inference."})
    
    # 训练标志，默认为 False
    training: bool = field(default=False, metadata={"help": "Benchmark training of model"})
    
    # 详细标志，默认为 False
    verbose: bool = field(default=False, metadata={"help": "Verbose memory tracing"})
    
    # 速度标志，默认为 True
    speed: bool = field(
        default=True,
        metadata={"help": "Whether to perform speed measurements. Speed measurements can be disabled via --no-speed."},
    )
    # 定义一个布尔类型的变量 memory，默认值为 True，用于控制是否进行内存测量
    memory: bool = field(
        default=True,
        metadata={
            "help": "Whether to perform memory measurements. Memory measurements can be disabled via --no-memory"
        },
    )
    # 定义一个布尔类型的变量 trace_memory_line_by_line，默认值为 False，用于控制是否逐行跟踪内存
    trace_memory_line_by_line: bool = field(default=False, metadata={"help": "Trace memory line by line"})
    # 定义一个布尔类型的变量 save_to_csv，默认值为 False，用于控制是否将结果保存到 CSV 文件
    save_to_csv: bool = field(default=False, metadata={"help": "Save result to a CSV file"})
    # 定义一个布尔类型的变量 log_print，默认值为 False，用于控制是否将所有打印语句保存到日志文件
    log_print: bool = field(default=False, metadata={"help": "Save all print statements in a log file"})
    # 定义一个布尔类型的变量 env_print，默认值为 False，用于控制是否打印环境信息
    env_print: bool = field(default=False, metadata={"help": "Whether to print environment information"})
    # 定义一个布尔类型的变量 multi_process，默认值为 True，用于控制是否使用多进程进行内存和速度测量
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
    # 定义一个字符串类型的变量 inference_time_csv_file，默认值为根据当前时间生成的推理时间结果的 CSV 文件名
    inference_time_csv_file: str = field(
        default=f"inference_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv."},
    )
    # 定义一个字符串类型的变量 inference_memory_csv_file，默认值为根据当前时间生成的推理内存结果的 CSV 文件名
    inference_memory_csv_file: str = field(
        default=f"inference_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv."},
    )
    # 定义一个字符串类型的变量 train_time_csv_file，默认值为根据当前时间生成的训练时间结果的 CSV 文件名
    train_time_csv_file: str = field(
        default=f"train_time_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving time results to csv for training."},
    )
    # 定义一个字符串类型的变量 train_memory_csv_file，默认值为根据当前时间生成的训练内存结果的 CSV 文件名
    train_memory_csv_file: str = field(
        default=f"train_memory_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving memory results to csv for training."},
    )
    # 定义一个字符串类型的变量 env_info_csv_file，默认值为根据当前时间生成的环境信息的 CSV 文件名
    env_info_csv_file: str = field(
        default=f"env_info_{round(time())}.csv",
        metadata={"help": "CSV filename used if saving environment information."},
    )
    # 定义一个字符串类型的变量 log_filename，默认值为根据当前时间生成的日志文件名
    log_filename: str = field(
        default=f"log_{round(time())}.csv",
        metadata={"help": "Log filename used if print statements are saved in log."},
    )
    # 定义一个整数类型的变量 repeat，默认值为 3，表示一个实验将运行的次数
    repeat: int = field(default=3, metadata={"help": "Times an experiment will be run."})
    # 定义一个布尔类型的变量 only_pretrain_model，默认值为 False，用于控制是否仅加载预训练模型权重而不加载定义在 `config.architectures` 中的模型
    only_pretrain_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Instead of loading the model as defined in `config.architectures` if exists, just load the pretrain"
                " model weights."
            )
        },
    )

    # 初始化方法，在实例化后发出警告，提示该类已被弃用
    def __post_init__(self):
        warnings.warn(
            f"The class {self.__class__} is deprecated. Hugging Face Benchmarking utils"
            " are deprecated in general and it is advised to use external Benchmarking libraries "
            " to benchmark Transformer models.",
            FutureWarning,
        )

    # 将实例序列化为 JSON 字符串的方法
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    # 属性
    @property
    # 返回模型名称列表
    def model_names(self) -> List[str]:
        # 如果模型列表为空，则抛出数值错误异常，提醒至少提供一个模型名称或模型标识符
        if len(self.models) <= 0:
            raise ValueError(
                "Please make sure you provide at least one model name / model identifier, *e.g.* `--models"
                " bert-base-cased` or `args.models = ['bert-base-cased']."
            )
        # 返回模型列表
        return self.models

    # 判断是否进行多进程处理
    @property
    def do_multi_processing(self):
        # 如果未启用多进程，则返回 False
        if not self.multi_process:
            return False
        # 如果在 TPU 上运行，因为当前不支持多进程，记录日志并返回 False
        elif self.is_tpu:
            logger.info("Multiprocessing is currently not possible on TPU.")
            return False
        # 否则返回 True，表示可以进行多进程处理
        else:
            return True
```