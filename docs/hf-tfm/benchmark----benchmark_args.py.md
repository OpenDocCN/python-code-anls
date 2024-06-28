# `.\benchmark\benchmark_args.py`

```
# 导入必要的库和模块
from dataclasses import dataclass, field
from typing import Tuple

# 导入工具函数和变量
from ..utils import (
    cached_property,
    is_torch_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
# 导入基准测试参数类
from .benchmark_args_utils import BenchmarkArguments

# 如果 Torch 可用，则导入 Torch 库
if is_torch_available():
    import torch

# 如果 Torch XLA 可用，则导入 Torch XLA 核心模块
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 PyTorch 基准测试参数类，继承自 BenchmarkArguments 类
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
        此 __init__ 方法用于向后兼容。在完全移除弃用参数后，可以简单删除这个类。
        """
        # 遍历所有弃用参数
        for deprecated_arg in self.deprecated_args:
            # 如果 kwargs 中包含该弃用参数
            if deprecated_arg in kwargs:
                # 获取对应的正向参数名
                positive_arg = deprecated_arg[3:]
                # 设置实例属性为相反的值
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                # 记录警告信息
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )

        # 设置 torchscript 属性，如果未提供则使用默认值 False
        self.torchscript = kwargs.pop("torchscript", self.torchscript)
        # 设置 torch_xla_tpu_print_metrics 属性，如果未提供则使用默认值 False
        self.torch_xla_tpu_print_metrics = kwargs.pop("torch_xla_tpu_print_metrics", self.torch_xla_tpu_print_metrics)
        # 设置 fp16_opt_level 属性，如果未提供则使用默认值 "O1"
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        
        # 调用父类的构造函数
        super().__init__(**kwargs)

    # 定义缓存属性的装饰器
    @cached_property
    # 设置设备初始化函数，返回一个 torch.device 对象和 GPU 数量
    def _setup_devices(self) -> Tuple["torch.device", int]:
        # 检查是否需要加载 torch 后端
        requires_backends(self, ["torch"])
        # 打印日志，标识正在设置 PyTorch 设备
        logger.info("PyTorch: setting up devices")
        # 如果不使用 CUDA，设备为 CPU，GPU 数量为 0
        if not self.cuda:
            device = torch.device("cpu")
            n_gpu = 0
        # 如果支持 Torch XLA，则设备为 TPU，GPU 数量为 0
        elif is_torch_xla_available():
            device = xm.xla_device()
            n_gpu = 0
        # 如果支持 Torch XPU，则设备为 XPU，获取 XPU 设备数量作为 GPU 数量
        elif is_torch_xpu_available():
            device = torch.device("xpu")
            n_gpu = torch.xpu.device_count()
        # 否则，默认使用 CUDA 设备，根据 CUDA 是否可用获取 GPU 数量
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        # 返回设备和 GPU 数量的元组
        return device, n_gpu

    @property
    def is_tpu(self):
        # 返回当前是否支持 Torch XLA 并且启用了 TPU
        return is_torch_xla_available() and self.tpu

    @property
    def device_idx(self) -> int:
        # 获取当前 CUDA 设备的索引
        requires_backends(self, ["torch"])
        # TODO(PVP): 目前仅支持单 GPU
        return torch.cuda.current_device()

    @property
    def device(self) -> "torch.device":
        # 返回设置的主设备对象（torch.device）
        requires_backends(self, ["torch"])
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        # 返回设置的 GPU 数量
        requires_backends(self, ["torch"])
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        # 返回是否至少有一个 GPU 可用
        return self.n_gpu > 0
```