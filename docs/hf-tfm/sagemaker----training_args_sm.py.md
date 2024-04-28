# `.\transformers\sagemaker\training_args_sm.py`

```
# 加载必要的库和模块
import importlib.util
import json
import os
import warnings
from dataclasses import dataclass, field
import torch
# 导入自定义的模块和函数
from ..training_args import TrainingArguments
from ..utils import cached_property, is_sagemaker_dp_enabled, logging

# 获取logger对象用于记录日志信息
logger = logging.get_logger(__name__)

# TODO: 重构SageMakerTrainer后应移到`utils`目录中

# 检查SageMaker是否支持模型并行训练
def is_sagemaker_model_parallel_available():
    # 从环境变量中获取SMP参数
    smp_options = os.getenv("SM_HP_MP_PARAMETERS", "{}")
    try:
        # 解析SMP参数并检查是否包含"partitions"字段，该字段是模型并行训练必需的
        smp_options = json.loads(smp_options)
        if "partitions" not in smp_options:
            return False
    except json.JSONDecodeError:
        return False

    # 从环境变量中获取Sagemaker特定框架参数
    mpi_options = os.getenv("SM_FRAMEWORK_PARAMS", "{}")
    try:
        # 解析MPI参数并检查"sagemaker_mpi_enabled"字段
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get("sagemaker_mpi_enabled", False):
            return False
    except json.JSONDecodeError:
        return False
    # 最后检查`smdistributed`模块是否存在
    return importlib.util.find_spec("smdistributed") is not None

# 如果SageMaker支持模型并行训练，则导入相关模块并初始化
if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp
    smp.init()

# 定义SageMaker训练参数类，继承自TrainingArguments
@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    # 定义mp_parameters字段，默认为空，用于SageMaker启动器发送mp特定参数，SageMakerTrainer中不受影响
    mp_parameters: str = field(
        default="",
        metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in SageMakerTrainer"},
    )

    # 后初始化函数
    def __post_init__(self):
        super().__post_init__()
        # 发出警告，提示`SageMakerTrainingArguments`已过时，并将在Transformers的v5版本中移除，建议使用`TrainingArguments`代替
        warnings.warn(
            "`SageMakerTrainingArguments` is deprecated and will be removed in v5 of Transformers. You can use "
            "`TrainingArguments` instead.",
            FutureWarning,
        )

    # 装饰器，将方法转换为属性，避免重复计算
    @cached_property
    # 设置设备，返回 torch.device
    def _setup_devices(self) -> "torch.device":
        # 输出日志信息
        logger.info("PyTorch: setting up devices")
        # 检查是否分布式训练环境并且分布式进程组已经初始化，但是 local_rank == -1，提示用户使用 Torch DDP 启动脚本
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
            logger.warning(
                "torch.distributed process group is initialized, but local_rank == -1. "
                "In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
            )
        # 如果设置了 no_cuda，则使用 CPU 设备
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        # 如果 SageMaker Model Parallel 可用，则使用当前进程的 local_rank 初始化设备
        elif is_sagemaker_model_parallel_available():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
        # 如果启用了 SageMaker Data Parallel，则初始化进程组并使用环境变量 "SMDATAPARALLEL_LOCAL_RANK" 的值作为 local_rank，初始化设备
        elif is_sagemaker_dp_enabled():
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401

            torch.distributed.init_process_group(backend="smddp", timeout=self.ddp_timeout_delta)
            self.local_rank = int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        # 如果 local_rank == -1，则根据 CUDA 是否可用，选择设备
        elif self.local_rank == -1:
            # 如果有多个 GPU，则使用 nn.DataParallel
            # 如果只想使用特定的一些 GPU，则使用 `CUDA_VISIBLE_DEVICES=0`
            # 显式地将 CUDA 设置为第一个（索引 0）CUDA 设备，否则 `set_device` 将会触发设备索引缺失的错误。索引 0 考虑了环境中可用的 GPU，因此 `CUDA_VISIBLE_DEVICES=1,2` 与 `cuda:0` 将使用环境中的第一个 GPU，即 GPU＃1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # 有时在我们到达这里之前还没有运行完 postinit 中的代码，所以只需检查是否不是默认值
            self._n_gpu = torch.cuda.device_count()
        # 在这里我们使用 torch.distributed
        # 初始化分布式后端，该后端将负责同步节点/ GPU
        else:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        # 如果设备类���是 "cuda"，则设置当前 CUDA 设备
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # 返回设备
        return device

    # 获取世界大小，如果可用 SageMaker Model Parallel，则返回 dp_size()，否则返回基类的 world_size
    @property
    def world_size(self):
        if is_sagemaker_model_parallel_available():
            return smp.dp_size()

        return super().world_size

    # 是否将模型放置在设备上，如果可用 SageMaker Model Parallel，则返回 False，否则返回 True
    @property
    def place_model_on_device(self):
        return not is_sagemaker_model_parallel_available()

    # 梯度累积时是否不同步
    @property
    def _no_sync_in_gradient_accumulation(self):
        return False
```