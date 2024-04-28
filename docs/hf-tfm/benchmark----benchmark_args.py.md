# `.\transformers\benchmark\benchmark_args.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 The HuggingFace Inc. team 和 NVIDIA CORPORATION 所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from dataclasses import dataclass, field
from typing import Tuple
from ..utils import cached_property, is_torch_available, is_torch_tpu_available, logging, requires_backends
from .benchmark_args_utils import BenchmarkArguments

# 如果 torch 可用，则导入 torch 模块
if is_torch_available():
    import torch

# 如果 torch TPU 可用，则导入 torch_xla.core.xla_model 模块
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 PyTorchBenchmarkArguments 类，继承自 BenchmarkArguments 类
@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):
    # 已弃用的参数列表
    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        # 处理已弃用的参数
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )

        # 处理新参数
        self.torchscript = kwargs.pop("torchscript", self.torchscript)
        self.torch_xla_tpu_print_metrics = kwargs.pop("torch_xla_tpu_print_metrics", self.torch_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        super().__init__(**kwargs)

    # 定义属性 torchscript，默认为 False，用于跟踪使用 torchscript 的模型
    torchscript: bool = field(default=False, metadata={"help": "Trace the models using torchscript"})
    # 定义属性 torch_xla_tpu_print_metrics，默认为 False，用于打印 Xla/PyTorch tpu 指标
    torch_xla_tpu_print_metrics: bool = field(default=False, metadata={"help": "Print Xla/PyTorch tpu metrics"})
    # 定义属性 fp16_opt_level，默认为 "O1"，用于 fp16: Apex AMP 优化级别选择
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    # 缓存属性
    @cached_property
    # 设置设备
    def _setup_devices(self) -> Tuple["torch.device", int]:
        # 要求后端为 Torch
        requires_backends(self, ["torch"])
        # 记录信息到日志
        logger.info("PyTorch: setting up devices")
        # 如果不使用 CUDA
        if not self.cuda:
            # 设备为 CPU
            device = torch.device("cpu")
            # GPU 数量为 0
            n_gpu = 0
        # 如果可用 TPU
        elif is_torch_tpu_available():
            # 使用 TPU 设备
            device = xm.xla_device()
            # GPU 数量为 0
            n_gpu = 0
        # 否则
        else:
            # 如果 CUDA 可用
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 获取 GPU 数量
            n_gpu = torch.cuda.device_count()
        # 返回设备和 GPU 数量
        return device, n_gpu

    # 判断是否为 TPU
    @property
    def is_tpu(self):
        # 返回 Torch TPU 是否可用且 TPU 是否开启
        return is_torch_tpu_available() and self.tpu

    # 获取设备索引
    @property
    def device_idx(self) -> int:
        # 要求后端为 Torch
        requires_backends(self, ["torch"])
        # 返回当前 CUDA 设备索引
        return torch.cuda.current_device()

    # 获取设备
    @property
    def device(self) -> "torch.device":
        # 要求后端为 Torch
        requires_backends(self, ["torch"])
        # 返回设置的设备
        return self._setup_devices[0]

    # 获取 GPU 数量
    @property
    def n_gpu(self):
        # 要求后端为 Torch
        requires_backends(self, ["torch"])
        # 返回设置的 GPU 数量
        return self._setup_devices[1]

    # 判断是否为 GPU
    @property
    def is_gpu(self):
        # 返回是否有 GPU
        return self.n_gpu > 0
```