# `.\sagemaker\training_args_sm.py`

```
# 导入必要的模块和库
import importlib.util  # 导入用于动态加载模块的模块
import json  # 导入处理 JSON 数据的模块
import os  # 导入与操作系统交互的模块
import warnings  # 导入用于处理警告的模块
from dataclasses import dataclass, field  # 导入用于创建数据类的装饰器和字段定义

import torch  # 导入 PyTorch 库

from ..training_args import TrainingArguments  # 从上级目录中导入训练参数类
from ..utils import cached_property, is_sagemaker_dp_enabled, logging  # 从上级目录中导入缓存属性装饰器、SageMaker DP 启用状态检查函数和日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象


# TODO: 在 SageMakerTrainer 重构后应移动到 `utils` 模块中


def is_sagemaker_model_parallel_available():
    # 从环境变量中获取 SageMaker 的模型并行参数
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # 解析 JSON 数据并检查是否包含 "partitions" 字段，模型并行需要此字段
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # 从环境变量中获取 SageMaker 的框架参数
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # 解析 JSON 数据并检查是否包含 "sagemaker_mpi_enabled" 字段
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False

    # 最后，检查是否存在 `smdistributed` 模块，以确认 SageMaker 是否支持模型并行
    return importlib.util.find_spec("smdistributed") is not None


# 如果 SageMaker 支持模型并行，则导入相应的模型并行库并进行初始化
if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp  # 导入 SageMaker 模型并行的 Torch 扩展库

    smp.init()  # 初始化 SageMaker 模型并行


@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in SageMakerTrainer"},
    )

    def __post_init__(self):
        super().__post_init__()
        # 发出警告，提示 `SageMakerTrainingArguments` 将在 Transformers v5 中被移除，建议使用 `TrainingArguments` 替代
        warnings.warn(
            "`SageMakerTrainingArguments` is deprecated and will be removed in v5 of Transformers. You can use "
            "`TrainingArguments` instead.",
            FutureWarning,
        )

    @cached_property
    # 设置设备
    def _setup_devices(self) -> "torch.device":
        # 打印日志信息
        logger.info("PyTorch: setting up devices")
        # 检查是否启用了torch分布式，并且本地进程的local_rank为-1
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            # 打印警告信息
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        # 如果禁用了CUDA
        if self.no_cuda:
            # 将设备设置为CPU
            device = torch.device("cpu")
            # GPU数量设为0
            self._n_gpu = 0
        # 如果支持SageMaker模型并行
        elif is_sagemaker_model_parallel_available():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            # GPU数量设为1
            self._n_gpu = 1
        # 如果启用了SageMaker分布式训练
        elif is_sagemaker_dp_enabled():
            # 导入SageMaker分布式训练模块
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401
            # 初始化进程组
            torch.distributed.init_process_group(backend="smddp", timeout=self.ddp_timeout_delta)
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        # 如果local_rank为-1
        elif self.local_rank == -1:
            # 如果n_gpu大于1，将使用nn.DataParallel。
            # 如果只想使用指定的GPU子集，可以使用`CUDA_VISIBLE_DEVICES=0`
            # 显式设置CUDA到第一个（索引0）CUDA设备，否则`set_device`会触发缺少设备索引的错误。
            # 索引0考虑了环境中可用的GPU，因此`CUDA_VISIBLE_DEVICES=1,2`与`cuda:0`将使用该环境中的第一个GPU，即GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # 有时在此之前尚未运行postinit中的行，因此只需检查我们不是默认值。
            self._n_gpu = torch.cuda.device_count()
        else:
            # 在这里，我们将使用torch分布式。
            # 初始化分布式后端，负责同步节点/GPU
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        # 如果设备类型为cuda
        if device.type == "cuda":
            # 设置当前使用的设备
            torch.cuda.set_device(device)

        # 返回设备
        return device

    @property
    # 获取world_size属性
    def world_size(self):
        # 如果支持SageMaker模型并行
        if is_sagemaker_model_parallel_available():
            # 返回并行大小
            return smp.dp_size()

        # 返回基类的world_size
        return super().world_size

    @property
    # 获取place_model_on_device属性
    def place_model_on_device(self):
        # 如果不支持SageMaker模型并行
        return not is_sagemaker_model_parallel_available()

    @property
    # 获取_no_sync_in_gradient_accumulation属性
    def _no_sync_in_gradient_accumulation(self):
        return False
```